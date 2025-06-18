# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZS-DeconvNet is a zero-shot learning tool for instant denoising and super-resolution in optical fluorescence microscopy. The project consists of three main components:

1. **Python/MATLAB Implementation** (`Python_MATLAB_Codes/`): Core training and inference code
2. **Fiji Plugin** (`Fiji_Plugin/`): ImageJ/Fiji plugin for GUI-based usage  
3. **Raw Data** (`Raw_Data/`): Test datasets for various microscopy modalities

## Architecture

### Python Implementation Structure
- `train_inference_python/`: Main Python codebase
  - `models/`: Neural network architectures (U-Net, RCAN3D variants)
  - `utils/`: Utility functions for data loading, augmentation, loss functions
  - Training scripts: `Train_ZSDeconvNet_2D.py`, `Train_ZSDeconvNet_3D.py`, etc.
  - Inference scripts: `Infer_2D.py`, `Infer_3D.py`

### MATLAB Data Generation
- `data_augment_recorrupt_matlab/`: MATLAB code for training data preparation
  - `GenData4ZS-DeconvNet/`: Standard data augmentation
  - `GenData4ZS-DeconvNet-SIM/`: SIM-specific data generation
  - `XxUtils/`: Common utility functions

### Model Types
The codebase supports multiple microscopy modalities:
- 2D/3D Wide-field microscopy
- 2D/3D Structured Illumination Microscopy (SIM)
- Confocal microscopy
- Lattice Light-Sheet Microscopy (LLSM)

## Common Development Commands

### Environment Setup
```bash
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt
```

### Training Models
```bash
# 2D training
bash train_demo_2D.sh

# 3D training  
bash train_demo_3D.sh

# SIM training
bash train_demo_2DSIM.sh
bash train_demo_3DSIM.sh
```

### Running Inference
```bash
# 2D inference
bash infer_demo_2D.sh

# 3D inference
bash infer_demo_3D.sh
```

### Converting Models for Fiji Plugin
```bash
cd Fiji_Plugin/TransferTFModelToPluginFormat
conda create -n tensorflow1 python=3.7
conda activate tensorflow1
pip install -r requirements.txt
bash tf_model2plugin_format.sh
```

## Key Technical Requirements

### Dependencies
- TensorFlow 2.5.0 for Python implementation
- TensorFlow 1.15.0 for Fiji plugin compatibility
- CUDA 11.4 + cuDNN for GPU acceleration
- MATLAB for data generation scripts

### Data Format Requirements
- Training data structure: `data_dir/folder/input/` and `data_dir/folder/gt/`
- PSF files: `.tif` format or `.mrc` format for OTF
- Image formats: `.tif` files for microscopy data
- PSF normalization: Divide by intensity summation before use

### Model Configuration
- 2D models: 128x128 patch size, batch size 4
- 3D models: 64x64x64 patch size, batch size 3
- Learning rates: 5e-5 (2D), 1e-4 (3D)
- Loss components: MSE/MAE + deconvolution + Hessian regularization

## Important File Locations

- Pre-trained models: Download from Google Drive links in README files
- Saved models directory: `Python_MATLAB_Codes/saved_models/` (excluded from git)
- Training data: Should be organized in `input/` and `gt/` subdirectories
- PSF files: Required for deconvolution loss calculation during training

## Development Notes

- Models saved during Python training can be directly used for inference
- For Fiji plugin usage, models must be converted to TensorFlow 1.x format
- PSF files must have correct dxy/dz values and be normalized
- Memory usage can be managed through tiling during inference
- The codebase supports both CPU and GPU execution