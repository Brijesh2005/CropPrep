"""Phase 4 — image model zoo (EfficientNetV2 / ConvNeXt) + embedding.

Canonical implementation shared between the local smoke environment and the
Kaggle training entry point.

Channel handling (documented in docs/PHASE_4_IMAGE_RESULTS.md):
    The input tensor has C = number of REAL Sentinel-2 channels discovered in
    the attached dataset (an index per channel, e.g. NDVI + EVI + GCI + MOISTURE
    => C=4). torchvision backbones are pretrained for 3-channel input, so the
    first convolution is adapted explicitly:

       * C == 3  -> pretrained RGB stem used as-is (genuine 3 spectral channels).
       * C != 3  -> the stem conv is replaced with a C-channel conv initialised
                    via the Kratzert style *mean* of the RGB-filter set
                    broadcast to C channels. This is a documented weight
                    initialisation, NOT silent channel duplication of fake bands.

    Never are channels duplicated to make the network run while real Sentinel-2
    channels are available.

Embedding contract (consumed by Phase 5 fusion, NOT implemented here):
    forward(x) returns (embed, logits).
    * EfficientNetV2-S : embed = 1280-d (adaptive global average pool).
    * ConvNeXt-Tiny    : embed =  768-d (avgpool + flatten before classifier).
    The embedding is saved together with the checkpoint; Phase 5 will consume
    it from models/image/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ARCH_BUILDERS = {}


def _register(name):
    def deco(fn):
        _ARCH_BUILDERS[name] = fn
        return fn
    return deco


@_register("efficientnet_v2_s")
def _build_efficientnet(pretrained: bool):
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
    return model, 1280


@_register("convnext_tiny")
def _build_convnext(pretrained: bool):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    return model, 768


def _stem_conv(model, arch: str) -> nn.Conv2d | None:
    """Return the first (3->k) convolution of a torchvision backbone, or None.

    ``model`` may be the full torchvision model (has ``.features``) or the
    features container itself; both are handled.
    """
    container = getattr(model, "features", None) or model
    try:
        if arch in ("efficientnet_v2_s", "convnext_tiny"):
            return container[0][0]
    except Exception:  # noqa: BLE001
        return None
    return None


def _replace_stem(model, arch: str, in_channels: int) -> nn.Conv2d | None:
    """Adapt the stem to in_channels; returns the OLD conv for mean-init reuse."""
    container = getattr(model, "features", None) or model
    if not hasattr(container, "__getitem__"):
        return None
    try:
        old = container[0][0] if arch in ("efficientnet_v2_s", "convnext_tiny") else None
    except Exception:  # noqa: BLE001
        return None
    if old is None or old.in_channels == in_channels:
        return old
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    container[0][0] = new
    return old


def _kratzert_init(new_conv: nn.Conv2d, old_conv: nn.Conv2d, in_channels: int) -> None:
    """Init a C-channel stem from the pretrained RGB filter set (mean init).

    weight_old: (out, 3, kh, kw) -> mean over the 3 input channels ->
                 (out, 1, kh, kw) -> broadcast to C channels.
    """
    with torch.no_grad():
        w = old_conv.weight.data.mean(dim=1, keepdim=True)  # (out,1,kh,kw)
        w = w / (in_channels > 0)
        new_conv.weight.data.copy_(w.expand(-1, in_channels, -1, -1))
        if new_conv.bias is not None and old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)


class ImageClassifier(nn.Module):
    """Backbone + classification head. forward(x) -> (embed, logits)."""

    def __init__(self, arch: str, in_channels: int, num_classes: int,
                 embed_dim: int, pretrained: bool, img_size: int):
        super().__init__()
        assert arch in _ARCH_BUILDERS, f"unknown arch {arch!r}"
        self.arch = arch
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.img_size = img_size

        backbone, default_embed = _ARCH_BUILDERS[arch](pretrained)
        self.features = backbone.features
        if embed_dim is None:
            embed_dim = default_embed
        self.embed_dim = embed_dim

        old = _replace_stem(self.features, arch, in_channels)
        if old is not None and old.in_channels != in_channels and pretrained:
            new = _stem_conv(self.features, arch)
            if new is not None:
                _kratzert_init(new, old, in_channels)

        if arch == "efficientnet_v2_s":
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            encoder = getattr(backbone, "classifier", None)
            in_feat = encoder[1].in_features if encoder is not None else 1280
            self.head = nn.Linear(in_feat, num_classes)
        elif arch == "convnext_tiny":
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_feat = 768
            self.head = nn.Linear(in_feat, num_classes)
        else:  # pragma: no cover - guarded by register
            raise ValueError(arch)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.features(x)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.features(x)
        out = self.avgpool(out)
        embed = torch.flatten(out, 1)
        logits = self.head(embed)
        return embed, logits


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

MODEL_CONFIG_KEYS = ["arch", "in_channels", "num_classes", "embed_dim",
                     "pretrained", "img_size", "stem_adaptation"]


def model_config(model: ImageClassifier) -> dict:
    return {
        "arch": model.arch,
        "in_channels": model.in_channels,
        "num_classes": model.num_classes,
        "embed_dim": model.embed_dim,
        "pretrained": model.pretrained,
        "img_size": model.img_size,
        "stem_adaptation": (
            "pretrained RGB stem used as-is"
            if model.in_channels == 3 else
            "Kratzert mean-init stem replacing the RGB conv"
        ),
    }


def build_model(config: dict, pretrained: bool | None = None) -> ImageClassifier:
    cfg = dict(config)
    if pretrained is not None:
        cfg["pretrained"] = pretrained
    return ImageClassifier(
        arch=cfg["arch"],
        in_channels=int(cfg["in_channels"]),
        num_classes=int(cfg["num_classes"]),
        embed_dim=(None if cfg.get("embed_dim") is None else int(cfg["embed_dim"])),
        pretrained=bool(cfg["pretrained"]),
        img_size=int(cfg["img_size"]),
    )


def save_model(model: ImageClassifier, path: Path, embed_dim: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model_config(model),
        "state_dict": model.state_dict(),
    }, path)


def load_model(path: Path, pretrained_for_weights: bool = True) -> tuple[ImageClassifier, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], pretrained=pretrained_for_weights)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["config"]