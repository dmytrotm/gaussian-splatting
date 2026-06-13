"""Script to download benchmark dataset(s)"""

import argparse
import os
import subprocess
from pathlib import Path

# dataset urls
urls = {
    "tandt": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip",
    "mipnerf360": [
        "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
    ],
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
    "tandt": "",
    "mipnerf360": "360_v2",
    "bilarf_data": "bilarf",
    "zipnerf": "zipnerf",
    "zipnerf_undistorted": "zipnerf_undistorted",
}


def download_and_extract(url: str, download_path: Path, extract_path: Path, progress_callback=None) -> None:
    download_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    import urllib.request
    last_percent = [-1]
    def reporthook(blocknum, blocksize, totalsize):
        if totalsize > 0:
            percent = int(blocknum * blocksize * 100 / totalsize)
            if percent != last_percent[0]:
                last_percent[0] = percent
                if progress_callback:
                    progress_callback(min(percent, 100))
                else:
                    print(f"[DOWNLOAD_PROGRESS] {min(percent, 100)}")

    try:
        urllib.request.urlretrieve(url, str(download_path), reporthook)
        print("File downloaded successfully.")
    except Exception as e:
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


def dataset_download(dataset: str, save_dir: Path, progress_callback=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_urls = urls[dataset]

    if isinstance(dataset_urls, list):
        for url in dataset_urls:
            url_file_name = Path(url).name
            extract_path = save_dir / dataset_rename_map[dataset]
            download_path = extract_path / url_file_name
            download_and_extract(url, download_path, extract_path, progress_callback)
    else:
        url_file_name = Path(dataset_urls).name
        extract_path = save_dir / dataset_rename_map[dataset]
        download_path = extract_path / url_file_name
        download_and_extract(dataset_urls, download_path, extract_path, progress_callback)


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
