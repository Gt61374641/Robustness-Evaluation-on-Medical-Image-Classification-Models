#!/usr/bin/env bash
# One-shot environment setup for an AutoDL (or any Linux) GPU instance.
#
# Usage (from the cloned repo root):
#   bash setup_cloud.sh            # install dependencies only
#   bash setup_cloud.sh --data     # install deps + download all three datasets
#
# Recommended: rent an instance whose base image already has PyTorch 2.x + CUDA
# 12.x, then this script only adds the project libs (no torch reinstall).
set -euo pipefail

# NOTE on location: the instance runs in a CN datacentre, so all downloads below
# use the instance's network, NOT your laptop's link — being in the UK does not
# slow them. We only need to speed up the CN datacentre's access to GitHub /
# HuggingFace / PyPI (international sites), which the mirrors below handle.

# --- AutoDL academic acceleration: proxies GitHub/HuggingFace from CN datacentres.
if [ -f /etc/network_turbo ]; then
  echo "==> Enabling AutoDL academic acceleration (/etc/network_turbo)..."
  source /etc/network_turbo
fi

# --- Fast pip mirror (Tsinghua) — installs from a domestic mirror, much faster.
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# --- HuggingFace mirror: timm pretrained weights download fast from CN datacentres.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "==> HF_ENDPOINT=$HF_ENDPOINT"
echo "==> pip mirror=$PIP_MIRROR"
echo "==> python: $(python --version 2>&1) ($(command -v python))"

# --- 1) PyTorch: reuse the base image's build; install only if missing. --------
if python -c "import torch" 2>/dev/null; then
  echo "==> torch present: $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
else
  echo "==> torch missing -> installing torch+torchvision from PyPI (bundled CUDA)."
  echo "    If this mismatches the instance CUDA, recreate the instance from a"
  echo "    PyTorch base image instead of installing torch here."
  pip install -i "$PIP_MIRROR" "torch>=2.0.0" "torchvision>=0.15.0"
fi

# --- 2) Project dependencies ---------------------------------------------------
echo "==> Installing project requirements..."
pip install -i "$PIP_MIRROR" -r requirements.txt

# --- 3) Sanity check -----------------------------------------------------------
python - <<'PY'
import torch, timm, art, sklearn
print("OK |",
      "torch", torch.__version__,
      "| cuda_available", torch.cuda.is_available(),
      "| device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
      "| timm", timm.__version__,
      "| ART", art.__version__)
PY

# --- 4) Configs (regenerate if missing) ---------------------------------------
if [ ! -f configs/oct2017_resnet18.yaml ]; then
  echo "==> Generating per-model configs..."
  python scripts/make_configs.py
fi

# --- 5) Optional: download datasets onto the instance (NOT uploaded) -----------
if [ "${1:-}" = "--data" ]; then
  echo "==> Downloading datasets (may require a Kaggle API token in ~/.kaggle/kaggle.json)..."
  python scripts/download_data.py --dataset malaria
  python scripts/download_data.py --dataset chest_xray_pneumonia
  python scripts/download_data.py --dataset oct2017
fi

echo ""
echo "==> Done."
echo "    Persist the HF mirror for future shells:"
echo "      echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc"
echo "    Then run e.g.:"
echo "      python scripts/train.py --config configs/malaria_resnet18.yaml"
