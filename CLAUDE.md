# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZS-DeconvNet is a zero-shot learning tool for instant denoising and super-resolution in optical fluorescence microscopy. The repository contains three main components:

1. **Python/MATLAB Implementation** (`Python_MATLAB_Codes/`) - Core training and inference code
2. **Fiji Plugin** (`Fiji_Plugin/`) - ImageJ/Fiji plugin for GUI-based usage
3. **Raw Data** (`Raw_Data/`) - Test datasets for various microscopy modalities

## Architecture

The system uses a two-stage U-Net architecture:
- **Stage 1**: Denoising network that removes noise from input images
- **Stage 2**: Deconvolution network that performs super-resolution on denoised output
- Both 2D and 3D variants are available
- Models support various microscopy types: wide-field, confocal, SIM, lattice light-sheet

## Key Components

### Python Training/Inference
- **Models**: `twostage_Unet.py` (2D), `twostage_Unet3D.py` (3D), `twostage_RCAN3D.py` (3D RCAN)
- **Training scripts**: `Train_ZSDeconvNet_*.py` for different modalities (2D, 3D, SIM variants)
- **Inference scripts**: `Infer_2D.py`, `Infer_3D.py`
- **Utilities**: Data loading, loss functions, augmentation in `utils/` directory

### MATLAB Data Processing
- **Data augmentation**: `GenData4ZS-DeconvNet/` for standard microscopy, `GenData4ZS-DeconvNet-SIM/` for SIM
- **PSF simulation**: Tools for generating theoretical PSFs when experimental ones aren't available

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
# 2D training example
bash train_demo_2D.sh

# 3D training example  
bash train_demo_3D.sh

# SIM training examples
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

### Model Conversion (Python to Fiji Plugin)
```bash
cd Fiji_Plugin/TransferTFModelToPluginFormat
conda create -n tensorflow1 python=3.7
conda activate tensorflow1
pip install -r requirements.txt
python TransferZSDeconv2DModelToPluginFormat.py  # for 2D models
python TransferZSDeconv3DModelToPluginFormat.py  # for 3D models
```

## Important Configuration Notes

- **PSF Requirements**: PSF files must have correct dxy and dz values, should be normalized by dividing by summation
- **Data Structure**: Training data should be organized as `data_dir/folder/input/` and `data_dir/folder/gt/`
- **Memory Management**: Use tiling for large images (adjust `num_seg_window_x/y`, `overlap_x/y`, `batch_size`)
- **GPU Setup**: Requires CUDA 11.4 + cuDNN for TensorFlow 2.5.0, or CUDA 10.1 + cuDNN 7.5.1 for Fiji plugin

## Key Parameters

### Training
- **Patch sizes**: 128x128 (2D), 64x64x64 (3D)
- **Learning rates**: 5e-5 (2D), 1e-4 (3D)
- **Batch sizes**: 4 (2D), 3 (3D)
- **Regularization**: Hessian weight 0.02 (2D), 0.1 (3D)

### Data Augmentation
- **Recorruption factors**: α=1-2, β₁=0.5-1.5, β₂=camera noise std
- **Augmentation counts**: 20,000 patches (2D), 10,000 patches (3D)

## File Structure Conventions

- Models saved as `.h5` files in `saved_models/`
- Training data in paired `input/` and `gt/` folders
- PSF files as `.tif` (PSF) or `.mrc` (OTF) format
- Test data organized by modality (WF2D, SIM2D, LLS3D, etc.)