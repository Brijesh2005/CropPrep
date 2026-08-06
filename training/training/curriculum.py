"""Five-stage curriculum training.

Curriculum training warms the model up progressively instead of training every
parameter from epoch one. Each stage unfreezes one component while the rest
stay frozen, so gradients first shape a single encoder and only later the
whole network:

    1. ``tabular``  — train the Tabular Encoder only.
    2. ``image``    — train the image encoders (NDVI / EVI + image fusion).
    3. ``temporal`` — train the Temporal Encoder.
    4. ``fusion``   — train the Fusion Engine (cross attention / gated fusion /
                      shared encoder).
    5. ``finetune`` — fine-tune the entire network.

Module ↔ model mapping (top-level :class:`~ai.models.cropfusion.
CropFusionModel` attributes):

* ``tabular``  → ``tab_encoder``
* ``image``    → ``ndvi_encoder``, ``evi_encoder``, ``image_fusion``
* ``temporal`` → ``temporal_transformer``, ``temporal_proj``
* ``fusion``   → ``fusion_engine`` (owns cross attention, gated fusion and the
  shared encoder)
* ``finetune`` → everything

Freezing semantics: frozen parameters get ``requires_grad=False`` **and** their
modules run in ``eval()`` mode (so BatchNorm statistics / Dropout stay inert);
the trainable scope stays in ``train()`` mode. :class:`CurriculumCallback`
applies the freeze on ``on_epoch_begin`` (on every rank, for DDP consistency)
and re-applies the eval-mode split after the trainer calls ``model.train()``
via the ``on_model_train_mode`` hook.

Stage scheduling is automatic: :meth:`Curriculum.stage_epochs` splits the
epoch budget across the active stages (``start_stage`` skips earlier stages,
which doubles as resume-from-any-stage; ``epochs_per_stage`` overrides
individual stages).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .config import CurriculumConfig
from .exceptions import TrainingRunError
from .interfaces import Callback

#: Canonical stage order (name ↔ position).
STAGE_ORDER: tuple[str, ...] = ("tabular", "image", "temporal", "fusion", "finetune")

#: The empty-string top-level module (the model root) carries no parameters of
#: its own and is skipped by the eval-mode split.
_ROOT_NAME = ""


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage (1-based number, name, trainable scope)."""

    number: int
    name: str
    #: Model attribute names (top-level ``CropFusionModel`` components) that
    #: are unfrozen during this stage. ``("__all__",)`` means the whole model.
    trainable: tuple[str, ...]
    description: str


CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(1, "tabular", ("tab_encoder",), "train the Tabular Encoder only"),
    CurriculumStage(
        2,
        "image",
        ("ndvi_encoder", "evi_encoder", "image_fusion"),
        "train the image encoders (NDVI / EVI + image fusion) only",
    ),
    CurriculumStage(
        3,
        "temporal",
        ("temporal_transformer", "temporal_proj"),
        "train the Temporal Encoder only",
    ),
    CurriculumStage(
        4,
        "fusion",
        ("fusion_engine",),
        "train the Fusion Engine (cross attention / gated fusion / shared encoder) only",
    ),
    CurriculumStage(5, "finetune", ("__all__",), "fine-tune the entire network"),
)

_STAGE_BY_NAME: dict[str, CurriculumStage] = {s.name: s for s in CURRICULUM_STAGES}


def stage_for(name_or_number: str | int) -> CurriculumStage:
    """Look up a :class:`CurriculumStage` by name or 1-based number."""
    if isinstance(name_or_number, int):
        for stage in CURRICULUM_STAGES:
            if stage.number == name_or_number:
                return stage
        raise TrainingRunError(
            f"unknown curriculum stage {name_or_number!r} (1..5)",
            detail=str(name_or_number),
        )
    try:
        return _STAGE_BY_NAME[name_or_number]
    except KeyError:
        raise TrainingRunError(
            f"unknown curriculum stage {name_or_number!r} "
            f"(expected one of {list(_STAGE_BY_NAME)})",
            detail=name_or_number,
        ) from None


def _top_level(module_name: str) -> str:
    """First segment of a ``named_modules`` / ``named_parameters`` name."""
    return module_name.split(".")[0]


class Curriculum:
    """Freeze / unfreeze a model's components per curriculum stage.

    Args:
        model: The :class:`~ai.models.cropfusion.CropFusionModel` to train.
        config: Validated :class:`CurriculumConfig`.
        num_epochs: Total training epochs (used to split the stage budget).
            Defaults to the sum of ``epochs_per_stage`` when given, else 5.
    """

    def __init__(
        self,
        model: nn.Module,
        config: CurriculumConfig | None = None,
        *,
        num_epochs: int | None = None,
    ) -> None:
        self.model = model
        self.config = config or CurriculumConfig()
        self._num_epochs = num_epochs
        self._current: CurriculumStage | None = None
        self._frozen_modules: list[str] = []

    # ------------------------------------------------------------------ #
    # Schedule
    # ------------------------------------------------------------------ #

    @property
    def num_epochs(self) -> int:
        if self._num_epochs is not None:
            return max(1, int(self._num_epochs))
        if self.config.epochs_per_stage:
            return max(1, sum(int(v) for v in self.config.epochs_per_stage.values()))
        return 5

    def _module_attributes(self, names: Sequence[str]) -> list[str]:
        """Filter ``names`` down to components that exist on the model."""
        existing: list[str] = []
        for name in names:
            if name == "__all__":
                return ["__all__"]
            if getattr(self.model, name, None) is not None:
                existing.append(name)
        return existing

    def active_stages(self) -> list[CurriculumStage]:
        """The stages actually run (``start_stage`` onward).

        Stages whose trainable components do not exist on the model are
        dropped (e.g. the ``image`` / ``fusion`` stages on a tabular-only
        model), so their epoch budget merges into the remaining stages.
        """
        return [
            stage
            for stage in CURRICULUM_STAGES[self.config.start_stage - 1 :]
            if self._module_attributes(stage.trainable)
        ]

    def stage_epochs(self) -> list[int]:
        """Per-stage epoch counts for the active stages (sum == ``num_epochs``)."""
        active = self.active_stages()
        active_names = {stage.name for stage in active}
        specified = {
            name: int(epochs)
            for name, epochs in (self.config.epochs_per_stage or {}).items()
            if name in active_names
        }
        if specified:
            leftover = max(self.num_epochs - sum(specified.values()), 0)
            unspecified = [stage.name for stage in active if stage.name not in specified]
            base, rem = divmod(leftover, max(len(unspecified), 1))
            counts: list[int] = []
            fill = 0
            for stage in active:
                if stage.name in specified:
                    counts.append(max(1, specified[stage.name]))
                else:
                    counts.append(max(1, base + (1 if fill < rem else 0)))
                    fill += 1
            return counts
        base, rem = divmod(self.num_epochs, max(len(active), 1))
        return [base + (1 if i < rem else 0) for i in range(len(active))]

    def stage_at(self, epoch: int) -> CurriculumStage:
        """Stage active at a 0-based epoch index."""
        if not self.config.enabled:
            return CURRICULUM_STAGES[-1]
        active = self.active_stages()
        budget = self.stage_epochs()
        acc = 0
        for stage, length in zip(active, budget):
            acc += length
            if epoch < acc:
                return stage
        return active[-1]

    # ------------------------------------------------------------------ #
    # Freeze / unfreeze
    # ------------------------------------------------------------------ #

    def apply_stage(self, stage: CurriculumStage | int | str) -> dict[str, Any]:
        """Apply a stage's freeze state to the model.

        Returns:
            A report dict with ``stage``, ``frozen`` (module names) and
            ``trainable`` (module names).
        """
        if isinstance(stage, (int, str)):
            stage = stage_for(stage)
        self._current = stage
        self._frozen_modules = []

        trainable = set(stage.trainable)
        if "__all__" in trainable:
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen_modules = []
            return {"stage": stage.name, "frozen": [], "trainable": ["(all)"]}

        for name, param in self.model.named_parameters():
            param.requires_grad = _top_level(name) in trainable

        # Report the top-level modules that actually hold parameters.
        frozen: set[str] = set()
        trainable_modules: set[str] = set()
        for module_name, module in self.model.named_modules():
            top = _top_level(module_name)
            if top == _ROOT_NAME:
                continue
            if not list(module.parameters(recurse=False)):
                continue  # container — its leaves report the top-level name
            (trainable_modules if top in trainable else frozen).add(top)
        frozen = sorted(frozen)
        trainable_modules = sorted(trainable_modules)
        self._frozen_modules = frozen
        return {
            "stage": stage.name,
            "frozen": frozen,
            "trainable": trainable_modules,
        }

    def apply_eval_mode(self) -> None:
        """Put frozen modules in ``eval()``, trainable ones back in ``train()``.

        Called after the trainer runs ``model.train()`` so the curriculum's
        per-module mode split is preserved (``model.train()`` flattens every
        submodule back to train mode).
        """
        if self._current is None:
            return
        if self._current.name == "finetune" or "__all__" in self._current.trainable:
            self.model.train()
            return
        trainable = set(self._current.trainable)
        for module_name, module in self.model.named_modules():
            top = _top_level(module_name)
            if top == _ROOT_NAME:
                continue
            if top in trainable:
                module.train()
            else:
                module.eval()

    @property
    def current_stage(self) -> CurriculumStage | None:
        return self._current


def build_curriculum(
    model: nn.Module,
    config: CurriculumConfig | None = None,
    *,
    num_epochs: int | None = None,
) -> Curriculum:
    """Build a :class:`Curriculum` for ``model``."""
    return Curriculum(model, config, num_epochs=num_epochs)


class CurriculumCallback(Callback):
    """Apply the curriculum's freeze state at the start of every epoch.

    Fires on **every** rank (``all_ranks = True``) so the parameter graph stays
    identical across processes in distributed training. Also implements the
    ``on_model_train_mode`` hook so the frozen-modules eval split survives the
    trainer's per-epoch ``model.train()``.
    """

    all_ranks: bool = True

    def __init__(self, curriculum: Curriculum) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.stages_log: list[dict[str, Any]] = []
        self.current_stage: CurriculumStage | None = None

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        stage = self.curriculum.stage_at(epoch)
        self.current_stage = stage
        info = self.curriculum.apply_stage(stage)
        info["epoch"] = epoch
        self.stages_log.append(dict(info))

    def on_model_train_mode(self) -> None:
        self.curriculum.apply_eval_mode()


__all__ = [
    "Curriculum",
    "CurriculumCallback",
    "CurriculumStage",
    "CURRICULUM_STAGES",
    "STAGE_ORDER",
    "build_curriculum",
    "stage_for",
]
