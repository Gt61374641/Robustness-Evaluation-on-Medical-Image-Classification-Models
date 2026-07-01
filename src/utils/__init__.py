# Torch-dependent utilities are imported lazily/optionally so that pure-plotting
# consumers (e.g. src.utils.plot_style on a machine without torch) can import
# this package without pulling in torch.
try:
    from .reproducibility import set_seed, save_config_snapshot, get_results_dir, get_checkpoint_path
except ModuleNotFoundError:  # torch not installed (local figure-only environment)
    pass
from .logger import get_logger
