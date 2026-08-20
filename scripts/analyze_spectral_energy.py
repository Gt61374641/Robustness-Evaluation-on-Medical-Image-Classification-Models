"""Radial spectral-energy profile of the three evaluation datasets.

Post-hoc diagnostic for the modality discussion (§5.2): measures where in the
spatial-frequency spectrum each dataset's pixel energy lives, under the same
224x224 [0, 1] preprocessing the models see. For each dataset a fixed,
deterministic sample of images is converted to grayscale, resized to 224x224,
mean-centred (DC removed), and Fourier-transformed; the per-image power
spectrum is binned by integer radial frequency (cycles per image, Nyquist =
112) and the fraction of AC energy above 28 and above 56 cycles per image is
recorded.

Sampling: Chest X-ray and OCT use their official test partitions. Malaria has
no on-disk test split (partitions are patient-grouped at runtime, §3.1), so
its sample is drawn from the full cell-image collection; image statistics are
a property of the collection, not of a partition.

    python scripts/analyze_spectral_energy.py
    # writes reports/thesis_evidence/spectral_energy.csv

Deterministic: files are sorted and subsampled with numpy seed 42.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "thesis_evidence" / "spectral_energy.csv"
SAMPLE_PER_DATASET = 256
SIZE = 224
SEED = 42
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASETS = {
    "chest_xray_pneumonia": ROOT / "data" / "chest_xray_pneumonia" / "test",
    "malaria": ROOT / "data" / "malaria" / "cell_images",
    "oct2017": ROOT / "data" / "oct2017" / "OCT2017" / "test",
}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def radial_energy_fractions(img: Image.Image) -> tuple[float, float]:
    """Fractions of AC spectral energy above 28 and 56 cycles/image."""
    arr = np.asarray(
        img.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float64
    ) / 255.0
    arr -= arr.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(yy - SIZE // 2, xx - SIZE // 2)
    total = power.sum()
    if total == 0:
        return 0.0, 0.0
    return float(power[r > 28].sum() / total), float(power[r > 56].sum() / total)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, root in DATASETS.items():
        paths = image_paths(root)
        if not paths:
            raise SystemExit(f"No images found under {root}")
        if len(paths) > SAMPLE_PER_DATASET:
            idx = rng.choice(len(paths), size=SAMPLE_PER_DATASET, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        hf28, hf56 = zip(*(radial_energy_fractions(Image.open(p)) for p in paths))
        rows.append(
            {
                "dataset": name,
                "n_images": len(paths),
                "hf_fraction_gt28_mean": np.mean(hf28),
                "hf_fraction_gt28_sd": np.std(hf28),
                "hf_fraction_gt56_mean": np.mean(hf56),
                "hf_fraction_gt56_sd": np.std(hf56),
            }
        )
        print(rows[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
