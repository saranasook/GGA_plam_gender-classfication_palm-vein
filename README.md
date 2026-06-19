# GGA_plam_gender-classfication_palm-vein

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
```

# 🛠️ Installation
1. Clone the Repository
git clone https://github.com/saranasook/GGA_plam_gender-classfication_palm-vein.git
```text
cd GGA_plam_gender-classfication_palm-vein
```

2. Install Dependencies
Ensure you have Python 3.8+ installed. You can install all required packages via pip:
```text
pip install -r requirements.txt
```

# 📊 Data Preparation
Our framework is evaluated using the public VERA Palm Vein Database. To prepare your data pipeline for training, complete the following steps:

Download the Dataset: Request and download the raw Near-Infrared (NIR) palm vein images from the official VERA dataset provider.

Generate CSV Metadata: Create train_data.csv and val_data.csv files. The dataset loader expects the CSV files to have no headers and be strictly mapped by column indices where:

      - Column 1 (index 1): Full absolute local file path to the target image (img_dir).

      - Column 2 (index 2): Gender string label (gender), denoted strictly as 'M' (Male) or 'F' (Female).


Example CSV Format:

0,/absolute/path/to/VERA/images/001_L.png,M
1,/absolute/path/to/VERA/images/002_F.png,F
2,/absolute/path/to/VERA/images/003_L.png,M


Note: Update the file paths inside train.py (train_dataset and val_dataset initializers) to point directly to your generated CSV files.


# 🚀 How to Use
1. Model Training & Evaluation
To start the model training loop using the GGA-DenseNet-161 paradigm over 200 epochs, verify your dataset paths inside gender_rain.py and run:
```text
python train.py
```
This script automatically evaluates validation metrics at every epoch, saves your optimal parameter configurations to best.pt, and outputs multi-case confusion and precision metrics for both demographic categories.


2. Single-Image Inference
To run a fast forward prediction pass on a single raw palm vein image file using a saved checkpoint weight array, use gender_inference.py:
```text
python gender_inference.py
```

Inside gender_inference.py usage example:
```text
result = predict_single_image(
    image_path="path/to/sample_palm.png",
    model_path="path/to/best.pt"
)
Output: {'probability': 0.9432, 'label_id': 1, 'gender': 'Female (F)'}
```


# 🔬 Core Methodology & Performance
Our GGA Module features parallel Spatial and Channel Attention gates deployed exclusively at the terminal feature bottleneck. This forces the model to ignore noisy peripheral hand contours and lock onto the central palm Region of Interest (ROI).

Peak Configuration: GGA-DenseNet-161

State-of-the-Art Accuracy: 94.39% on the VERA Database

Key Visual Proof: Grad-CAM activations demonstrate precise spatial localization matching internal vascular geometry.


# 📜 Reference & Citation
If you use this code, the GGA module architecture, or find our benchmarks helpful in your biometric research, please cite our work as presented in the workshop proceedings of the conference:
@inproceedings{yourname2026global,
  title={Global Gated Attention for Robust Gender Classification in Palm Vein Biometrics},
  author={Your Name and Co-Authors},
  booktitle={Proceedings of the Workshop on Empathic AI: Face, Gesture, and Accessibility Technologies, held in conjunction with the 20th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2026)},
  year={2026},
  month={May},
  publisher={IEEE}
}

For inquiries regarding NECTEC biometric collaborations, please contact sorn.soo@nectec.or.th



