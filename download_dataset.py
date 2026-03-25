"""Script to download benchmark dataset(s)"""

import argparse
import os
import subprocess
from pathlib import Path

# dataset urls
urls = {
    "mipnerf360": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "mipnerf360_extra": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
    "bilarf_data": "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip",
    "zipnerf": [
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/berlin.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/london.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/nyc.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf/alameda.zip",
    ],
    "zipnerf_undistorted": [
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
    ],
}

# rename maps
dataset_rename_map = {
    "mipnerf360": "360_v2",
    "mipnerf360_extra": "360_v2",
    "bilarf_data": "bilarf",
    "zipnerf": "zipnerf",
    "zipnerf_undistorted": "zipnerf_undistorted",
}


def download_and_extract(url: str, download_path: Path, extract_path: Path) -> None:
    download_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # download
    download_command = [
        "curl",
        "-L",
        "-o",
        str(download_path),
        url,
    ]
    try:
        subprocess.run(download_command, check=True)
        print("File downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return

    # if .zip
    if Path(url).suffix == ".zip":
        if os.name == "nt":  # Windows doesn't have 'unzip' but 'tar' works
            extract_command = [
                "tar",
                "-xvf",
                str(download_path),
                "-C",
                str(extract_path),
            ]
        else:
            extract_command = [
                "unzip",
                "-o",
                str(download_path),
                "-d",
                str(extract_path),
            ]
    # if .tar
    else:
        extract_command = [
            "tar",
            "-xvzf",
            str(download_path),
            "-C",
            str(extract_path),
        ]

    # extract
    try:
        subprocess.run(extract_command, check=True)
        os.remove(download_path)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")


def dataset_download(dataset: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_urls = urls[dataset]

    if isinstance(dataset_urls, list):
        for url in dataset_urls:
            url_file_name = Path(url).name
            extract_path = save_dir / dataset_rename_map[dataset]
            download_path = extract_path / url_file_name
            download_and_extract(url, download_path, extract_path)
    else:
        url_file_name = Path(dataset_urls).name
        extract_path = save_dir / dataset_rename_map[dataset]
        download_path = extract_path / url_file_name
        download_and_extract(dataset_urls, download_path, extract_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark dataset(s)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mipnerf360",
        choices=list(urls.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Directory to save dataset",
    )
    args = parser.parse_args()

    dataset_download(dataset=args.dataset, save_dir=Path(args.save_dir))
