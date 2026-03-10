# 3D Gaussian Splatting: Quick Start Guide

This repository has been modernized to run seamlessly on modern hardware (Python 3.12+, PyTorch 2.6.0+, CUDA 12.4+) and includes an integrated web-based viewer.

Follow these step-by-step instructions to get the project running from scratch on a new machine.

## 1. Clone the Repository
Clone this repository to your local machine and navigate into the directory:
```bash
git clone https://github.com/dmytrotm/gaussian-splatting.git
cd gaussian-splatting
```

## 2. Set Up the Python Environment
Create a new virtual environment using Python 3.12 and activate it:
```bash
# Create the virtual environment
python3.12 -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# (Alternatively, for Windows Command Prompt)
# .venv\Scripts\activate.bat
```

## 3. Install Dependencies
Install the required Python packages (including PyTorch for CUDA 12) from the updated `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 4. Install Custom CUDA Modules
This project relies on three custom C++/CUDA modules for rendering and math. Install them using pip:
```bash
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

## 5. Download a Dataset
We have provided a robust script to automatically download the NeRF synthetic "Lego" dataset to test the pipeline.
```bash
python download_dataset.py
```
*This will download and extract the dataset into the `data/` folder.*

## 6. Run Training & View Live in Browser
Start the training script and enable the modern web viewer. 

```bash
python train.py -s data/lego --port 6009 --web_viewer
```

Once the script starts, it will print: `Starting Viser web viewer on http://localhost:6009`

- **Local Machine:** Open your web browser and navigate to `http://localhost:6009`.
- **VS Code (Remote SSH):** Go to the "Ports" tab at the bottom, click "Add Port", type `6009`, and press Enter. Then click the globe icon to open the forwarded port in your browser.

You will see the 3D scene construct itself in real-time as the training sequence processes!

---

### (Optional) Legacy SIBR Viewer
If you prefer the original C++ TCP client (SIBR) instead of the web browser viewer, simply run the training script without the `--web_viewer` flag:
```bash
python train.py -s data/lego --port 6009
```
