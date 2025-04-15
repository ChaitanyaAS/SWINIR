# SwinIR-Like Super-Resolution on DIV2K Dataset

This project implements a simplified version of the SwinIR (Swin Transformer for Image Restoration) model for single-image super-resolution (SISR). The model is trained on the DIV2K dataset and evaluated using PSNR and SSIM metrics.

---

##  Features

- Uses PyTorch for model development and training
- Simplified SwinIR-style CNN architecture (not full Swin Transformer)
- DIV2K dataset loader with random patch extraction and downscaling 
- Image super-resolution with upscaling factor 4×
- PSNR and SSIM evaluation metrics
- Visualization of low-res, super-res, and high-res images

---

##  Directory Structure

project/ │ ├── DIV2K/ │ ├── DIV2K_train_HR/ │ └── DIV2K_valid_HR/ │ ├── swinir_sr.py # Main training and evaluation script ├── swinir_div2k.pth # Trained model (after training) └── README.md # This file


---

##  Requirements

Install the dependencies using pip:

pip install numpy opencv-python torch torchvision matplotlib tqdm scikit-image


## Install Dataset
Download the DIV2K dataset (HR images):
Training: DIV2K_train_HR
Validation: DIV2K_valid_HR

 ## Training
 python swinir_sr.py

## Evaluation
PSNR: 17.87 dB
SSIM: 0.5048

## Visualization
The script includes a function to visualize:
Low-resolution input
Super-resolved output
Ground-truth high-resolution image

## Model Overview
The architecture is inspired by SwinIR but implemented as a simple CNN:
Convolutional feature extractor
Residual-like layers with batch norm and ReLU
PixelShuffle-based upsampling

## Notes
Only the HR images are used (LR images are generated on-the-fly via downscaling).
The model uses L1 loss.
Training can be further improved with more epochs and a deeper architecture.

## Author
Created by Chaitanya A S 
Feel free to reach out for questions or collaboration ideas!
