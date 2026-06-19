# GGA_plam_gender-classfication_plam-vein

# Global Gated Attention (GGA) for Robust Gender Classification in Palm Vein Biometrics

This repository contains the official PyTorch implementation for our framework **GGA-CNN**, which introduces a **Global Gated Attention (GGA)** module at the terminal bottleneck of deep networks to mitigate demographic bias and accurately classify gender from Near-Infrared (NIR) palm vein images.

---

## 📌 Repository Structure

```text
├── gender_model.py          # Model architectures (DenseNet161_GGA_Binary, etc.)
├── gender_train.py                 # Core training loop and evaluation scripts
├── gender_inference.py             # Single-image inference pipeline script
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation


🛠️ Installation
1. Clone the Repository
git clone https://github.com/saranasook/GGA_plam_gender-classfication_plam-vein.git
cd GGA_plam_gender-classfication_plam-vein


2. Install Dependencies
Ensure you have Python 3.8+ installed. You can install all required packages via pip:

pip install -r requirements.txt

📊 Data Preparation
Our framework is evaluated using the public VERA Palm Vein Database. To prepare your data pipeline for training, complete the following steps:

Download the Dataset: Request and download the raw Near-Infrared (NIR) palm vein images from the official VERA dataset provider.

Generate CSV Metadata: Create train_data.csv and val_data.csv files. The dataset loader expects the CSV files to have no headers and be strictly mapped by column indices where:

Column 1 (index 1): Full absolute local file path to the target image (img_dir).

Column 2 (index 2): Gender string label (gender), denoted strictly as 'M' (Male) or 'F' (Female).

