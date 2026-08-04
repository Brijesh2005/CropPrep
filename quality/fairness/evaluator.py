"""Fairness evaluator — group parity statistics and verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .config import FairnessConfig
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    regression_metrics,
    roc_auc,
)


@dataclass
class GroupResult:
    """Metrics for one sensitive-attribute group."""

    group: str
    attribute: str
    support: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FairnessVerdict:
    """A single threshold comparison."""

    metric: str
    value: float
    threshold: float
    status: str  # compliant | at_risk | violating

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FairnessResult:
    """Full fairness report for a task."""

    task: str
    attribute: str
    groups: list[GroupResult] = field(default_factory=list)
    verdicts: list[FairnessVerdict] = field(default_factory=list)
    overall_status: str = "compliant"
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FairnessEvaluator:
    """Compare model outcomes across sensitive groups."""

    def __init__(self, config: FairnessConfig | None = None) -> None:
        self.config = config or FairnessConfig()

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        groups: Mapping[str, Sequence[Any]],
        *,
        y_proba: Sequence[float] | None = None,
        y_pred_regression: Sequence[float] | None = None,
        task: str = "classification",
    ) -> FairnessResult:
        """Evaluate fairness for each sensitive attribute in ``groups``.

        Args:
            y_true: Ground-truth binary labels (or continuous for regression).
            y_pred: Predicted binary labels (or continuous for regression).
            groups: ``{attribute: values-per-sample}``.
            y_proba: Predicted probability of the positive class (for
                calibration / AUC).
        """
        results = []
        for attribute, values in groups.items():
            results.append(
                self._evaluate_attribute(
                    attribute,
                    values,
                    y_true,
                    y_pred,
                    y_proba=y_proba,
                    y_pred_regression=y_pred_regression,
                    task=task,
                )
            )
        return self._aggregate(results)

    # ------------------------------------------------------------------ #

    def _evaluate_attribute(
        self,
        attribute: str,
        values: Sequence[Any],
        y_true: Sequence[int],
        y_pred: Sequence[int],
        *,
        y_proba: Sequence[float] | None,
        y_pred_regression: Sequence[float] | None,
        task: str,
    ) -> FairnessResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        values = np.asarray(values)
        unique = sorted(set(str(v) for v in values))
        groups: list[GroupResult] = []
        for label in unique:
            mask = np.asarray([str(v) == label for v in values])
            if mask.sum() < self.config.min_group_size:
                groups.append(
                    GroupResult(
                        group=label,
                        attribute=attribute,
                        support=int(mask.sum()),
                        metrics={"status": "insufficient_data"},
                    )
                )
                continue
            if task == "regression" and y_pred_regression is not None:
                metrics = regression_metrics(y_true[mask], y_pred_regression[mask])
            else:
                metrics = classification_metrics(y_true[mask], y_pred[mask])
                if y_proba is not None:
                    metrics["ece"] = expected_calibration_error(
                        y_true[mask], np.asarray(y_proba)[mask], bins=self.config.calibration_bins
                    )["ece"]
                    metrics["roc_auc"] = roc_auc(y_true[mask], np.asarray(y_proba)[mask])
            groups.append(GroupResult(group=label, attribute=attribute, support=int(mask.sum()), metrics=metrics))

        return self._verdicts_for(attribute, groups, task)

    # ------------------------------------------------------------------ #

    def _verdicts_for(
        self, attribute: str, groups: list[GroupResult], task: str
    ) -> FairnessResult:
        valid = [g for g in groups if "status" not in g.metrics]
        cfg = self.config
        verdicts: list[FairnessVerdict] = []
        if len(valid) < 2:
            result = FairnessResult(task=task, attribute=attribute, groups=groups)
            result.overall_status = "insufficient_groups"
            return result

        def _range_ratio(metric: str) -> tuple[float, float, float]:
            values = [g.metrics[metric] for g in valid]
            lo, hi = min(values), max(values)
            return lo, hi, hi - lo

        if task == "classification":
            # Statistical parity (base-rate parity).
            _, _, spread = _range_ratio("base_rate")
            verdicts.append(
                FairnessVerdict(
                    metric="statistical_parity",
                    value=spread,
                    threshold=cfg.statistical_parity_max,
                    status=_status(spread, cfg.statistical_parity_max, cfg.severity_multiplier),
                )
            )
            base_rates = [g.metrics["base_rate"] for g in valid]
            lo, hi = min(base_rates), max(base_rates)
            ratio = lo / hi if hi > 0 else 1.0
            verdicts.append(
                FairnessVerdict(
                    metric="disparate_impact",
                    value=ratio,
                    threshold=cfg.disparate_impact_min,
                    status=_status_inverse(ratio, cfg.disparate_impact_min, cfg.severity_multiplier),
                )
            )
            tpr_lo, _, tpr_spread = _range_ratio("tpr")
            fpr_lo, _, fpr_spread = _range_ratio("fpr")
            verdicts.append(
                FairnessVerdict(
                    metric="equalized_odds",
                    value=max(tpr_spread, fpr_spread),
                    threshold=cfg.equalized_odds_max,
                    status=_status(max(tpr_spread, fpr_spread), cfg.equalized_odds_max, cfg.severity_multiplier),
                )
            )
            verdicts.append(
                FairnessVerdict(
                    metric="equal_opportunity",
                    value=tpr_spread,
                    threshold=cfg.equal_opportunity_max,
                    status=_status(tpr_spread, cfg.equal_opportunity_max, cfg.severity_multiplier),
                )
            )
            acc_lo, _, acc_spread = _range_ratio("accuracy")
            verdicts.append(
                FairnessVerdict(
                    metric="accuracy_parity",
                    value=acc_spread,
                    threshold=cfg.accuracy_parity_max,
                    status=_status(acc_spread, cfg.accuracy_parity_max, cfg.severity_multiplier),
                )
            )
            if all("ece" in g.metrics for g in valid):
                _, _, ece_spread = _range_ratio("ece")
                verdicts.append(
                    FairnessVerdict(
                        metric="calibration_parity",
                        value=ece_spread,
                        threshold=cfg.calibration_parity_max,
                        status=_status(ece_spread, cfg.calibration_parity_max, cfg.severity_multiplier),
                    )
                )
        else:
            # Regression: bias and error parity.
            _, _, mae_spread = _range_ratio("mae")
            verdicts.append(
                FairnessVerdict(
                    metric="error_parity",
                    value=mae_spread,
                    threshold=cfg.accuracy_parity_max,
                    status=_status(mae_spread, cfg.accuracy_parity_max, cfg.severity_multiplier),
                )
            )
            signed = [g.metrics["signed_bias"] for g in valid]
            bias_spread = abs(max(signed) - min(signed))
            verdicts.append(
                FairnessVerdict(
                    metric="signed_bias_parity",
                    value=bias_spread,
                    threshold=cfg.accuracy_parity_max,
                    status=_status(bias_spread, cfg.accuracy_parity_max, cfg.severity_multiplier),
                )
            )

        summary = {
            "num_groups": len(groups),
            "min_group_support": min((g.support for g in groups), default=0),
            "worst_metric": max(verdicts, key=lambda v: _violation_ratio(v)).metric,
        }
        overall = _overall_status(verdicts)
        return FairnessResult(
            task=task,
            attribute=attribute,
            groups=groups,
            verdicts=verdicts,
            overall_status=overall,
            summary=summary,
        )

    def _aggregate(self, results: list[FairnessResult]) -> FairnessResult:
        if not results:
            return FairnessResult(task="classification", attribute="none")
        statuses = {r.overall_status for r in results}
        if "violating" in statuses:
            overall = "violating"
        elif "at_risk" in statuses:
            overall = "at_risk"
        else:
            overall = "compliant"
        return FairnessResult(
            task=results[0].task,
            attribute=", ".join(r.attribute for r in results),
            groups=[g for r in results for g in r.groups],
            verdicts=[v for r in results for v in r.verdicts],
            overall_status=overall,
            summary={"attributes": [r.attribute for r in results]},
        )


def _status(value: float, threshold: float, multiplier: float) -> str:
    if value <= threshold:
        return "compliant"
    if value <= threshold * multiplier:
        return "at_risk"
    return "violating"


def _status_inverse(value: float, threshold: float, multiplier: float) -> str:
    if value >= threshold:
        return "compliant"
    if value >= threshold / multiplier:
        return "at_risk"
    return "violating"


def _violation_ratio(verdict: FairnessVerdict) -> float:
    if verdict.status == "compliant":
        return 0.0
    if verdict.metric == "disparate_impact":
        return verdict.threshold / max(verdict.value, 1e-9)
    return verdict.value / max(verdict.threshold, 1e-9)


def _overall_status(verdicts: list[FairnessVerdict]) -> str:
    if any(v.status == "violating" for v in verdicts):
        return "violating"
    if any(v.status == "at_risk" for v in verdicts):
        return "at_risk"
    return "compliant"
