# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZS-DeconvNet is a zero-shot learning tool for instant denoising and super-resolution in optical fluorescence microscopy. The repository provides three main implementations:

1. **Python/MATLAB Implementation** (`Python_MATLAB_Codes/`) - Core training and inference
2. **Fiji Plugin** (`Fiji_Plugin/`) - ImageJ/Fiji plugin for easy GUI usage  
3. **PyTorch Implementation** (`PyTorch_Deconv/`) - Alternative PyTorch version

## Architecture

### Core Models
- **2D ZS-DeconvNet**: Two-stage U-Net architecture for 2D deconvolution (`models/twostage_Unet.py`)
- **3D ZS-DeconvNet**: 3D variant using RCAN architecture (`models/twostage_RCAN3D.py`, `models/twostage_Unet3D.py`)
- Both models use a two-stage approach: first stage for denoising, second stage for deconvolution

### Key Components
- **Data Augmentation**: MATLAB scripts in `data_augment_recorrupt_matlab/` for training data generation
- **Loss Functions**: Custom loss combining MSE/MAE, deconvolution loss, and regularization terms (Hessian, TV, L1)
- **PSF Integration**: Point Spread Function handling for deconvolution calculations

## Development Commands

### Python Environment Setup
```bash
# Create conda environment
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet

# Install dependencies (TensorFlow version)
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt

# Install CUDA support
conda install cudatoolkit==11.3.1
conda install cudnn==8.2.1
```

### Training Models
```bash
# 2D training
cd Python_MATLAB_Codes/train_inference_python
bash train_demo_2D.sh

# 3D training
bash train_demo_3D.sh

# SIM-specific training
bash train_demo_2DSIM.sh
bash train_demo_3DSIM.sh
```

### Inference
```bash
# 2D inference
bash infer_demo_2D.sh

# 3D inference
bash infer_demo_3D.sh
```

### Fiji Plugin Conversion
```bash
# Convert Python model to Fiji plugin format
conda create -n tensorflow1 python=3.7
conda activate tensorflow1
cd Fiji_Plugin/TransferTFModelToPluginFormat
pip install -r requirements.txt

# For 2D models
python TransferZSDeconv2DModelToPluginFormat.py

# For 3D models  
python TransferZSDeconv3DModelToPluginFormat.py
```

## Key Parameters

### Training Parameters
- **2D Models**: patch_size=128, batch_size=4, epochs=250, lr=5e-5
- **3D Models**: patch_size=64, batch_size=3, epochs=100, lr=1e-4
- **Loss weights**: Hessian regularization (0.02 for 2D, 0.1 for 3D)

### Data Requirements
- Input images in `data_dir/folder/input/`
- Ground truth in `data_dir/folder/gt/`
- PSF files in .tif or .mrc format
- Sampling intervals (dx, dy, dz) must match PSF

### Model Files
- TensorFlow 2.5.0 models saved as .h5 weights
- Fiji plugin models in .zip format (SaveModelBundle)
- Pre-trained models available via Google Drive links

## Data Structure
- `Raw_Data/`: Original microscopy data (2D/3D, various modalities)
- `saved_models/`: Pre-trained models organized by modality
- Training data generated via MATLAB augmentation scripts
- PSF files required for deconvolution loss calculation

## Important Notes
- TensorFlow GPU 2.5.0 required for training
- Fiji plugin uses TensorFlow-Java 1.15.0
- PSF normalization critical - divide by sum before use
- 3D models require proper z-axis dimension mapping
- Memory management via tiling for large images