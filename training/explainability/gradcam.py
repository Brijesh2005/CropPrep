"""GradCAM-family image explainer for the NDVI / EVI encoders.

Supports GradCAM, GradCAM++ (Chattopadhay et al., 2018), EigenCAM (Muhammad &
Yeamin, 2021) and LayerCAM (Jiang et al., 2021). Heatmaps are computed from the
last spatial convolutional feature map of a timm backbone and its gradient
w.r.t. the explained output.

Produces per-timestep heatmaps for NDVI and EVI, overlays, and PNG / NumPy
exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .config import CamConfig
from .exceptions import CamError
from .interfaces import CamMethod
from .utils import (
    find_last_spatial_conv,
    normalize_heatmap,
    resize_heatmap,
    single_sample_batch,
    to_numpy,
)


# --------------------------------------------------------------------------- #
# CAM computations
# --------------------------------------------------------------------------- #


def _cam_gradcam(activations: torch.Tensor, gradients: torch.Tensor, relu: bool = True) -> torch.Tensor:
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [.., C, 1, 1]
    cam = (weights * activations).sum(dim=1)  # [.., H, W]
    return torch.relu(cam) if relu else cam


def _cam_gradcam_plusplus(activations: torch.Tensor, gradients: torch.Tensor, relu: bool = True) -> torch.Tensor:
    g = torch.relu(gradients)
    eps = 1e-8
    # alpha_c = G_c^2 / (2 G_c^2 + sum_ij A_ij * G_c^3)
    denom = 2.0 * g.sum(dim=(2, 3), keepdim=True) + (
        activations * g * g * g
    ).sum(dim=(2, 3), keepdim=True)
    alpha = g * g / denom.clamp_min(eps)
    weights = (alpha * g).sum(dim=(2, 3), keepdim=True)  # [.., C, 1, 1]
    cam = (weights * activations).sum(dim=1)
    return torch.relu(cam) if relu else cam


def _cam_eigencam(activations: torch.Tensor, gradients: torch.Tensor, relu: bool = True) -> torch.Tensor:
    del gradients  # EigenCAM needs no gradient
    batch = activations.shape[:-3]
    flat = activations.flatten(0, -4)  # [B, C, H, W]
    b, c, h, w = flat.shape
    mat = flat.view(b, c, h * w)
    # Top eigenvector of mat @ mat^T (PCA of channels).
    gram = mat @ mat.transpose(1, 2)
    _, _, v = torch.linalg.svd(gram)
    vec = v[..., 0, :]  # [B, C]
    cam = torch.einsum("bc,bcw->bw", vec, mat.view(b, c, h * w))  # [B, H*W]
    cam = cam.view(b, h, w)
    return torch.relu(cam) if relu else cam


def _cam_layercam(activations: torch.Tensor, gradients: torch.Tensor, relu: bool = True) -> torch.Tensor:
    cam = (torch.relu(gradients) * activations).sum(dim=1)
    return torch.relu(cam) if relu else cam


class GradCAM(CamMethod):
    name = "gradcam"

    def weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        return gradients.mean(dim=(2, 3))


class GradCAMPlusPlus(CamMethod):
    name = "gradcam++"

    def weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        g = torch.relu(gradients)
        denom = 2.0 * g.sum(dim=(2, 3), keepdim=True) + (
            activations * g * g * g
        ).sum(dim=(2, 3), keepdim=True)
        alpha = g * g / denom.clamp_min(1e-8)
        return (alpha * g).sum(dim=(2, 3))


class EigenCAM(CamMethod):
    name = "eigencam"

    def weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        raise CamError("EigenCAM is computed directly; use cam_eigencam")

    def compute(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        return _cam_eigencam(activations, gradients, self.relu)


class LayerCAM(CamMethod):
    name = "layercam"

    def weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        # Spatial weights: [.., C, H, W]
        return torch.relu(gradients)


_CAM_METHODS = {
    "gradcam": _cam_gradcam,
    "gradcam++": _cam_gradcam_plusplus,
    "eigencam": _cam_eigencam,
    "layercam": _cam_layercam,
}


def compute_cam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    method: str = "gradcam++",
    relu: bool = True,
) -> torch.Tensor:
    """Compute a class-activation map from activations and gradients."""
    if method not in _CAM_METHODS:
        raise CamError(f"unknown CAM method {method!r}", detail=sorted(_CAM_METHODS))
    return _CAM_METHODS[method](activations, gradients, relu=relu)


# --------------------------------------------------------------------------- #
# Image explainer
# --------------------------------------------------------------------------- #


class ImageExplainer:
    """Per-timestep GradCAM heatmaps for the NDVI / EVI encoders.

    Args:
        model: The trained :class:`~ai.models.cropfusion.CropFusionModel`.
        config: Validated :class:`CamConfig`.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: CamConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or CamConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not getattr(model, "use_image", False):
            raise CamError("ImageExplainer requires an image branch")
        self._targets: dict[str, nn.Conv2d] = {}

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    def _hook_target(self, index: str) -> tuple[str, nn.Conv2d]:
        encoder = getattr(self.model, f"{index}_encoder", None)
        if encoder is None:
            raise CamError(f"model has no {index.upper()} encoder")
        if index not in self._targets:
            override = self.config.target_layer
            if override:
                match = None
                for name, module in encoder.backbone.named_modules():
                    if override in name and isinstance(module, nn.Conv2d):
                        match = module
                        break
                if match is None:
                    raise CamError(
                        f"target layer {override!r} not found in {index} backbone"
                    )
                self._targets[index] = match
            else:
                self._targets[index] = find_last_spatial_conv(
                    encoder.backbone, tuple(encoder.input_size)
                )[1]
        return index, self._targets[index]

    # ------------------------------------------------------------------ #
    # Core explanation
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample: Mapping[str, torch.Tensor],
        *,
        index: str = "ndvi",
        timestep: int | None = None,
        kind: str = "crop",
        class_index: int | None = None,
    ) -> dict[str, Any]:
        """Per-timestep heatmaps for one NDVI or EVI sequence.

        Returns ``{"heatmaps": np.ndarray [T, H, W], "timesteps": int,
        "activations_shape", "gradients_norm"}``.
        """
        _, conv = self._hook_target(index)
        batch = single_sample_batch(sample, self.device)

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_mod: nn.Module, _args: Any, output: torch.Tensor) -> None:
            activations["a"] = output.detach()

        def backward_hook(_mod: nn.Module, _gi: Any, go: Any) -> None:
            gradients["g"] = go[0].detach()

        fh = conv.register_forward_hook(forward_hook)
        bh = conv.register_full_backward_hook(backward_hook)
        try:
            out = self.model(batch)
            target = self._target_scalar(out, kind, class_index)
            self.model.zero_grad()
            target.backward()
        finally:
            fh.remove()
            bh.remove()

        if "a" not in activations or "g" not in gradients:
            raise CamError("gradient did not reach the CAM target layer")
        A = activations["a"]  # [T, C, H', W']
        G = gradients["g"]  # [T, C, H', W']
        timesteps = int(A.shape[0])

        cams = compute_cam(A, G, method=self.config.method, relu=self.config.relu)
        cams = cams.detach().cpu().numpy()  # [T, H', W']

        # Resize to the patch resolution.
        encoder = getattr(self.model, f"{index}_encoder", None)
        patch_hw = encoder.input_size  # (H, W)
        heatmaps = np.stack(
            [normalize_heatmap(resize_heatmap(cams[t], patch_hw)) for t in range(timesteps)]
        )

        selected = timestep if timestep is not None else int(np.argmax(heatmaps.mean(axis=(1, 2))))
        return {
            "heatmaps": heatmaps,
            "timestep": selected,
            "timesteps": timesteps,
            "activations_shape": tuple(A.shape),
            "gradients_norm": float(G.norm().item()),
        }

    def _target_scalar(
        self, out: Any, kind: str, class_index: int | None
    ) -> torch.Tensor:
        if kind == "crop":
            logits = out.crop_logits
            if logits is None:
                raise CamError("model has no crop head")
            cls = class_index if class_index is not None else int(logits[0].argmax().item())
            return logits[0, cls]
        pred = out.yield_pred
        if pred is None:
            raise CamError("model has no yield head")
        return pred[0, 0]

    # ------------------------------------------------------------------ #
    # Overlay / export
    # ------------------------------------------------------------------ #

    def overlay(
        self,
        sample: Mapping[str, torch.Tensor],
        result: Mapping[str, Any],
        *,
        index: str = "ndvi",
        timestep: int | None = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay a heatmap onto the patch image (``[H, W, 3]`` RGB)."""
        heatmap = result["heatmaps"]
        t = timestep if timestep is not None else int(result["timestep"])
        patch = sample[index][t, 0].numpy()  # [H, W]
        hm = heatmap[t]

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(self.config.colormap)
        colored = cmap(hm)[..., :3]  # [H, W, 3]
        if patch.ndim == 2:
            base = np.repeat(patch[:, :, None], 3, axis=2)
            lo, hi = float(base.min()), float(base.max())
            base = (base - lo) / (hi - lo + 1e-12)
        else:
            base = patch
        overlay_img = (1.0 - alpha) * base + alpha * colored
        return np.clip(overlay_img, 0.0, 1.0)

    def export_png(
        self,
        result: Mapping[str, Any],
        path: str | Path,
        *,
        timestep: int | None = None,
    ) -> Path:
        """Save a heatmap as a PNG."""
        import matplotlib.pyplot as plt

        heatmap = result["heatmaps"]
        t = timestep if timestep is not None else int(result["timestep"])
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap[t], cmap=self.config.colormap, interpolation="bilinear")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out

    def export_numpy(
        self, result: Mapping[str, Any], path: str | Path
    ) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, result["heatmaps"])
        return out
