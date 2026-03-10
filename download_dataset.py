import argparse
import os
import zipfile
import subprocess
from pathlib import Path
from typing import Literal

# We use huggingface_hub to reliably download the NeRF synthetic dataset
# which fits in the limited disk space of this environment.
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call(["pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

def download_and_extract(dataset: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset == "nerf_synthetic":
        print("Downloading nerf_synthetic.zip from HuggingFace...")
        zip_path = hf_hub_download(
            repo_id="XayahHina/nerf_synthetic",
            filename="nerf_synthetic.zip",
            repo_type="dataset",
            local_dir=str(save_dir)
        )
        
        print(f"Downloaded to {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
            
        print(f"Extraction complete! Dataset available at: {save_dir / 'nerf_synthetic'}")
        
    else:
        print(f"Skip downloading huge dataset {dataset} to prevent disk out-of-space.")
        print("Please use 'nerf_synthetic' for testing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to download benchmark dataset(s)")
    parser.add_argument("--dataset", type=str, default="nerf_synthetic", choices=["nerf_synthetic", "mipnerf360", "bilarf_data", "zipnerf"])
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), "data"))
    args = parser.parse_args()

    download_and_extract(dataset=args.dataset, save_dir=Path(args.save_dir))
