from .chest_xray import get_chest_xray_loaders
from .retinal import get_retinal_loaders
from .malaria import get_malaria_loaders
from .transforms import get_transforms

DATASET_REGISTRY = {
    "chest_xray_pneumonia": get_chest_xray_loaders,
    "oct2017": get_retinal_loaders,
    "malaria": get_malaria_loaders,
}

NUM_CLASSES = {
    "chest_xray_pneumonia": 2,
    "oct2017": 4,
    "malaria": 2,
}


def get_dataloaders(cfg):
    """Get train/val/test dataloaders based on config."""
    dataset_name = cfg["data"]["dataset"]
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_REGISTRY.keys())}")
    loader_fn = DATASET_REGISTRY[dataset_name]
    return loader_fn(cfg)
