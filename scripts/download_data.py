"""Unified data download script for the three project datasets.

Usage:
    python scripts/download_data.py --dataset chest_xray_pneumonia  # Kaggle (Kermany Pneumonia)
    python scripts/download_data.py --dataset oct2017               # Kaggle (Kermany OCT2017)
    python scripts/download_data.py --dataset malaria               # Kaggle (NIH Malaria Cell Images)

All three are pulled from Kaggle. You will need:
    1. pip install kaggle
    2. A Kaggle API token at ~/.kaggle/kaggle.json
       Get it from https://www.kaggle.com/settings -> API -> Create New Token
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_kaggle_dataset(dataset_slug: str, output_dir: str, competition: bool = False):
    """Download a dataset (or competition) from Kaggle.

    Args:
        dataset_slug: Either a `user/name` dataset slug or, for competitions,
            a competition name.
        output_dir: Directory to extract files into.
        competition: If True, treat ``dataset_slug`` as a Kaggle competition name
            and use ``competition_download_files``.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        print("Then place your API token at ~/.kaggle/kaggle.json")
        print("Get it from: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_slug} to {output_dir}...")
    if competition:
        api.competition_download_files(dataset_slug, path=output_dir)
    else:
        api.dataset_download_files(dataset_slug, path=output_dir, unzip=True)
    print(f"Done! Files saved to {output_dir}/")
    if competition:
        print(
            "NOTE: competition downloads come as a single zip; you will need to "
            "unzip it manually and arrange files so that train.csv and "
            "train/image/*.jpg sit under the dataset directory."
        )


KAGGLE_DATASETS = {
    "chest_xray_pneumonia": {
        "slug": "paultimothymooney/chest-xray-pneumonia",
        "subdir": "chest_xray_pneumonia",
        "competition": False,
    },
    "oct2017": {
        "slug": "paultimothymooney/kermany2018",
        "subdir": "oct2017",
        "competition": False,
    },
    "malaria": {
        # NIH Malaria Cell Images (Rajaraman et al.); extracts to cell_images/{Parasitized,Uninfected}.
        "slug": "iarunava/cell-images-for-detecting-malaria",
        "subdir": "malaria",
        "competition": False,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download medical image datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(KAGGLE_DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument("--data-dir", default="./data", help="Directory to save data (default: ./data)")
    args = parser.parse_args()

    info = KAGGLE_DATASETS[args.dataset]
    output_dir = os.path.join(args.data_dir, info["subdir"])
    download_kaggle_dataset(info["slug"], output_dir, competition=info["competition"])


if __name__ == "__main__":
    main()
