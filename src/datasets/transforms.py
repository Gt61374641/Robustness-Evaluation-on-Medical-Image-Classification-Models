"""Unified image transforms for medical image classification."""

from torchvision import transforms


def get_transforms(img_size: int = 224, is_training: bool = True):
    """Get image transforms for training or evaluation.

    Args:
        img_size: Target image size (default 224 for most timm models).
        is_training: If True, apply data augmentation.

    Returns:
        torchvision.transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            # NOTE: Do NOT normalize here — ART expects pixel values in [0, 1].
            # Normalization is handled inside the model wrapper or ART classifier.
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
