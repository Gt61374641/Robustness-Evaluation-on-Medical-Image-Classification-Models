"""Model factory using timm.

The current experimental scope keeps DenseNet121 as the single baseline model.
The classification head is replaced to match the target number of classes.
"""

import timm
import torch
import torch.nn as nn


SUPPORTED_MODELS = {
    "densenet121": "densenet121",
}


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

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


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
