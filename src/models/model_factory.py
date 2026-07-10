"""Create timm models with preprocessing consistent across the pipeline.

Images remain in pixel space (``[0, 1]``) for ART. ImageNet normalization is
therefore applied at the model boundary for training, attacks and features.
"""

import warnings

import timm
import torch
import torch.nn as nn


# ResNet complexity ladder (one family, monotonically increasing complexity) for
# the complexity-vs-robustness study. Approx. parameter counts: resnet18 ~11.7M,
# resnet34 ~21.8M, resnet50 ~25.6M, resnet101 ~44.5M, resnet152 ~60.2M.
#
# Architecture comparison (chest primary dataset): two non-ResNet architectures
# parameter-matched to ResNet-50 (~25.6M). Pretrained tags are pinned to
# ImageNet-1k-only SUPERVISED weights so the comparison isolates architecture,
# not pretraining data: DeiT-S IS the ViT-S/16 architecture trained on IN1k only
# (the plain vit_small default tag was pretrained on IN21k); ConvNeXt-T uses the
# original Facebook IN1k weights.
SUPPORTED_MODELS = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    "deit_small": "deit_small_patch16_224.fb_in1k",   # ViT-S/16, ~22.1M
    "convnext_tiny": "convnext_tiny.fb_in1k",         # modern CNN, ~28.6M
}


class NormalizedModel(nn.Module):
    """Apply a timm backbone's expected input normalization."""

    def __init__(self, backbone: nn.Module, mean, std):
        super().__init__()
        self.backbone = backbone
        self.normalization_enabled = True
        self.register_buffer(
            "input_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return x
        return (x - self.input_mean) / self.input_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))

    def forward_features(self, x: torch.Tensor):
        return self.backbone.forward_features(self.normalize(x))

    def forward_head(self, x, *args, **kwargs):
        return self.backbone.forward_head(x, *args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load both wrapped checkpoints and legacy raw-timm checkpoints."""
        if state_dict and not any(key.startswith("backbone.") for key in state_dict):
            # Checkpoints created before the wrapper were trained directly on
            # [0, 1] pixels. Preserve their original inference semantics; new
            # training runs save prefixed keys and keep normalization enabled.
            self.normalization_enabled = False
            warnings.warn(
                "Loading a legacy checkpoint trained without input normalization. "
                "Retrain it before comparing against newly trained models.",
                UserWarning,
                stacklevel=2,
            )
            return self.backbone.load_state_dict(state_dict, strict=strict, assign=assign)
        self.normalization_enabled = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create a model from timm with the specified number of output classes.

    Args:
        model_name: Public name (key in SUPPORTED_MODELS) or any valid timm model name.
        num_classes: Number of output classes for the classification head.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        PyTorch model ready for fine-tuning.
    """
    timm_name = SUPPORTED_MODELS.get(model_name, model_name)

    # timm.list_models() enumerates *architecture* names; pretrained tags after
    # the dot (e.g. ".fcmae_ft_in22k_in1k") are not listed, so we validate the
    # architecture part only.
    arch_name = timm_name.split(".")[0]
    if not timm.list_models(arch_name):
        raise ValueError(
            f"Model '{model_name}' (timm name: '{timm_name}') not found. "
            f"Available models: {list(SUPPORTED_MODELS.keys())}"
        )

    backbone = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    pretrained_cfg = getattr(backbone, "pretrained_cfg", {})
    mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
    std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
    return NormalizedModel(backbone, mean, std)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load model weights from a checkpoint file."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    return model


def get_model_info(model_name: str) -> dict:
    """Get model metadata (parameter count, etc.)."""
    timm_name = SUPPORTED_MODELS.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=False, num_classes=1000)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": model_name,
        "timm_name": timm_name,
        "num_params": num_params,
        "num_trainable_params": num_trainable,
        "num_params_M": round(num_params / 1e6, 1),
    }
