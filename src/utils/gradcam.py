"""Grad-CAM for clean vs adversarial attention comparison.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image
regions that most influence a model's prediction for a target class. In this
project it is a *supporting* analysis: comparing clean and adversarial Grad-CAM
maps shows whether an attack shifts the model's attention away from the lesion,
which qualitatively explains the deep-feature differences that detectors exploit.

The implementation supports both:
- channels-first activations (B, C, H, W) — CNN conv/BN layers (ResNet, DenseNet);
- channels-last activations (B, H, W, C) — LayerNorm layers in ConvNeXt / Swin.

The activation layout is auto-detected from the target layer type (LayerNorm =>
channels-last) and can be overridden via ``channels_last``.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM heatmap generator.

    Args:
        model: a PyTorch model in (or to be set to) eval mode.
        target_layer: the module whose activations/gradients define the CAM
            (typically the last conv layer for CNNs, or the final norm for
            ConvNeXt/Swin).
        channels_last: force the activation layout. If None, inferred as True
            for nn.LayerNorm target layers and False otherwise.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, channels_last: bool | None = None):
        self.model = model
        self.target_layer = target_layer
        self.channels_last = (
            isinstance(target_layer, nn.LayerNorm) if channels_last is None else channels_last
        )
        self._activations = None
        self._gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self._activations = output
        # Capture the gradient flowing into this activation during backward.
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad

    def __call__(self, image: torch.Tensor, class_idx=None) -> torch.Tensor:
        """Compute the Grad-CAM heatmap.

        Args:
            image: input tensor (B, C, H, W) in [0, 1].
            class_idx: target class. None => use the model's predicted class.
                May be an int or a (B,) tensor/list of per-sample classes.

        Returns:
            Heatmap tensor (B, 1, H, W) in [0, 1], on CPU, detached.
        """
        self.model.eval()
        was_4d = image.dim() == 4
        if not was_4d:
            image = image.unsqueeze(0)
        # requiring grad on the input guarantees the graph builds even if the
        # model parameters happen to be frozen.
        image = image.detach().clone().requires_grad_(True)

        logits = self.model(image)
        batch = logits.shape[0]

        if class_idx is None:
            target = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            target = torch.full((batch,), class_idx, dtype=torch.long, device=logits.device)
        else:
            target = torch.as_tensor(class_idx, dtype=torch.long, device=logits.device)

        score = logits.gather(1, target.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations/gradients. "
                               "Check that target_layer is part of the forward pass.")

        if self.channels_last:
            # (B, H, W, C): weights over spatial dims, weighted sum over channels.
            weights = gradients.mean(dim=(1, 2), keepdim=True)        # (B, 1, 1, C)
            cam = (weights * activations).sum(dim=3)                  # (B, H, W)
        else:
            # (B, C, H, W): weights over spatial dims, weighted sum over channels.
            weights = gradients.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
            cam = (weights * activations).sum(dim=1)                  # (B, H, W)

        cam = F.relu(cam).unsqueeze(1)                                # (B, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # Per-sample min-max normalisation to [0, 1].
        cam = cam.detach()
        b = cam.shape[0]
        flat = cam.view(b, -1)
        cmin = flat.min(dim=1).values.view(b, 1, 1, 1)
        cmax = flat.max(dim=1).values.view(b, 1, 1, 1)
        cam = (cam - cmin) / (cmax - cmin).clamp_min(1e-12)

        return cam.cpu()

    def close(self):
        """Remove the forward hook."""
        self._fwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Panel rendering                                                             #
# --------------------------------------------------------------------------- #
def _to_hwc(image: torch.Tensor) -> np.ndarray:
    """(C, H, W) or (1, C, H, W) tensor in [0, 1] -> (H, W, C) numpy."""
    img = image.detach().cpu()
    if img.dim() == 4:
        img = img[0]
    return np.clip(img.permute(1, 2, 0).numpy(), 0.0, 1.0)


def _to_hw(cam: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) / (1, H, W) / (H, W) tensor -> (H, W) numpy."""
    c = cam.detach().cpu()
    while c.dim() > 2:
        c = c[0]
    return c.numpy()


def _save_panel(out_stem, clean, adv, clean_cam, adv_cam, true_label=None,
                clean_pred=None, adv_pred=None, formats=("svg", "png", "pdf")):
    """Render a clean-vs-adversarial Grad-CAM comparison panel.

    Layout (2 rows x 2 cols): top = clean image / clean Grad-CAM overlay;
    bottom = adversarial image / adversarial Grad-CAM overlay.

    Args:
        out_stem: output path stem (extension ignored).
        clean, adv: (C, H, W) image tensors in [0, 1].
        clean_cam, adv_cam: Grad-CAM heatmaps (broadcastable to (H, W)).
        true_label/clean_pred/adv_pred: optional strings/ints for titles.
        formats: output formats to write.

    Returns:
        Dict mapping format -> Path of the written files (always includes svg, png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Editable SVG text so the panel satisfies the publication contract.
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"

    clean_img, adv_img = _to_hwc(clean), _to_hwc(adv)
    clean_h, adv_h = _to_hw(clean_cam), _to_hw(adv_cam)

    fig, axes = plt.subplots(2, 2, figsize=(4.0, 4.2))

    def show(ax, img, title, heat=None):
        ax.imshow(img)
        if heat is not None:
            ax.imshow(heat, cmap="jet", alpha=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ct = "Clean" if clean_pred is None else f"Clean (pred: {clean_pred})"
    at = "Adversarial" if adv_pred is None else f"Adversarial (pred: {adv_pred})"
    show(axes[0, 0], clean_img, ct)
    show(axes[0, 1], clean_img, "Clean Grad-CAM", clean_h)
    show(axes[1, 0], adv_img, at)
    show(axes[1, 1], adv_img, "Adversarial Grad-CAM", adv_h)

    if true_label is not None:
        fig.suptitle(f"True label: {true_label}", fontsize=8)

    fig.tight_layout()

    base = Path(out_stem)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        kwargs = {"dpi": 300} if fmt in ("png", "tiff", "tif", "jpg", "jpeg") else {}
        fig.savefig(out, bbox_inches="tight", **kwargs)
        outputs[fmt] = out
    plt.close(fig)
    return outputs
