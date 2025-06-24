# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZS-DeconvNet is a self-supervised deep learning tool for simultaneous denoising and super-resolution in optical fluorescence microscopy. The project implements zero-shot learning that doesn't require paired training data.

## Architecture

The codebase consists of three main implementations:

1. **Python/TensorFlow**: Primary research implementation in `Python_MATLAB_Codes/train_inference_python/`
2. **MATLAB**: Data preprocessing and PSF generation in `Python_MATLAB_Codes/data_augment_recorrupt_matlab/`
3. **Fiji Plugin**: End-user plugin for ImageJ/Fiji in `Fiji_Plugin/`
4. **PyTorch**: Alternative implementation in `PyTorch_Deconv/` (newer, pure deconvolution only)

## Environment Setup

### Python Environment
```bash
conda activate zs-deconvnet_pytorch
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt
```

### CUDA Dependencies
```bash
conda install cudatoolkit==11.3.1
conda install cudnn==8.2.1
```

## Core Implementation: train_inference_python Directory

### Main Training Scripts

#### `Train_ZSDeconvNet_2D.py`
- **Purpose**: 2D training for wide-field microscopy
- **Model**: Two-stage U-Net (denoising + deconvolution)
- **Key Parameters**:
  - `--conv_block_num`: U-Net depth (default 4)
  - `--conv_num`: Convolutions per block (default 3)
  - `--upsample_flag`: Enable 2x super-resolution (0/1)
  - `--iterations`: Training steps (typically 50000)
  - `--start_lr`: Learning rate (typically 5e-5)

#### `Train_ZSDeconvNet_2DSIM.py`
- **Purpose**: 2D SIM (Structured Illumination Microscopy) training
- **Differences**: Includes `augment_sim_img.py` for SIM-specific augmentation
- **Data augmentation**: Random cropping, 8-fold rotation/flipping

#### `Train_ZSDeconvNet_3D.py`
- **Purpose**: 3D volumetric training (wide-field, confocal, LLS)
- **Models**: Supports both `twostage_Unet3D` and `twostage_RCAN3D`
- **Key Parameters**:
  - `--insert_xy`: XY padding (default 8)
  - `--insert_z`: Z padding (default 2)
  - `--iterations`: Typically 10000 for 3D
  - `--batch_size`: Usually 1-2 for memory constraints

#### `Train_ZSDeconvNet_3DSIM.py`
- **Purpose**: 3D SIM training with enhanced loss functions
- **Architecture**: RCAN3D variants (standard, compact, compact2)

### Inference Scripts

#### `Infer_2D.py`
- **Purpose**: 2D inference with patch-based processing
- **Key Parameters**:
  - `--num_seg_window_x/y`: Image segmentation for large images
  - `--overlap_x/y`: Overlap between patches (avoid artifacts)
  - `--bs`: Batch size tuple for multiple inputs
- **Memory Management**: Automatic segmentation for large images

#### `Infer_3D.py`
- **Purpose**: 3D volumetric inference
- **Segmentation**: 3D patch processing with overlap handling
- **Output**: Saves both denoised and deconvolved results

### Model Architecture Details

#### `models/twostage_Unet.py` (2D)
```python
# Two-stage architecture:
# Stage 1: Denoising (self-supervised)
# Stage 2: Deconvolution (PSF-based)
# 
# Key components:
# - Encoder: 4 downsampling blocks (conv_block_num=4)
# - Decoder: 4 upsampling blocks with skip connections
# - Optional: 2x upsampling for super-resolution
# - Output cropping: Remove padding artifacts
```

#### `models/twostage_Unet3D.py` (3D)
```python
# 3D version differences:
# - Conv3D, MaxPooling3D, UpSampling3D operations
# - Z-axis pooling: (2,2,1) to preserve axial resolution
# - Separate XY and Z padding parameters
```

#### `models/twostage_RCAN3D.py` (3D RCAN)
```python
# Residual Channel Attention Network variants:
# - RCAN3D: Standard version
# - RCAN3D_SIM: Enhanced for SIM with ReLU regularization
# - RCAN3D_SIM_compact/compact2: Lightweight versions
# 
# Key components:
# - RCAB: Residual Channel Attention Block
# - ResidualGroup: Multiple RCAB blocks (n_RCAB=2-4)
# - Channel Attention: Global pooling + squeeze-excitation
```

### Utilities Directory

#### `utils/loss.py` - Comprehensive Loss Functions
```python
# PSF-based losses:
create_psf_loss()      # 2D PSF loss with regularization
create_psf_loss_3D()   # 3D volumetric version

# Self-supervised loss:
create_NBR2NBR_loss()  # Neighbor2Neighbor approach

# Regularization terms:
# - TV (Total Variation): Edge-preserving smoothness
# - Hessian: Second-order derivative penalty  
# - L1: Sparsity promotion

# PSF utilities:
cal_psf_2d/3d()       # PSF calculation from OTF
psf_estimator_2d/3d() # PSF parameter estimation
```

#### `utils/data_loader.py` - Data Loading
```python
# DataLoader with normalization options:
# norm_flag=0: /65535 normalization
# norm_flag=1: Percentile normalization (robust)
# norm_flag=2: Max normalization
```

#### `utils/utils.py` - General Utilities
```python
prctile_norm()  # Percentile normalization (0-100 percentile)
read_mrc()      # MRC file reader for OTF files
```

#### `utils/augment_sim_img.py` - SIM Data Augmentation
```python
aug_sim_img_2D/3D()  # SIM-specific augmentation
augment_img()        # 8 augmentation modes (rotation, flipping)
```

### Shell Script Demos

#### Training Demos
- `train_demo_2D.sh`: Wide-field 2D with detailed parameter explanations
- `train_demo_2DSIM.sh`: SIM-specific 2D training
- `train_demo_3D.sh`: 3D volumetric examples
- `train_demo_3DSIM.sh`: 3D SIM training

#### Inference Demos  
- `infer_demo_2D.sh`: Examples for WF, SIM modalities
- `infer_demo_3D.sh`: LLS, Confocal, LLS-SIM examples

### Training Workflow

1. **Data Organization**:
   ```
   data_dir/folder/
   ├── input/     # Corrupted training images
   └── gt/        # Ground truth images
   ```

2. **PSF/OTF Setup**:
   - PSF: `.tif` format (psf_src_mode=1)
   - OTF: `.mrc` format (psf_src_mode=2)

3. **Parameter Configuration**:
   ```bash
   python Train_ZSDeconvNet_2D.py \
     --otf_or_psf_path 'path/to/psf.mrc' \
     --data_dir 'path/to/training/data' \
     --folder 'training_folder_name' \
     --test_images_path 'path/to/test.tif' \
     --psf_src_mode 2
   ```

4. **Loss Function Weights**:
   - `--denoise_loss_weight`: Balance denoising vs deconvolution (0.5)
   - `--TV_rate`: Total variation regularization (0-0.1)
   - `--Hess_rate`: Hessian regularization (0.02-0.1)
   - `--l1_rate`: L1 sparsity (typically 0)

### Inference Workflow

1. **Model Loading**: Load trained weights from `.h5` files
2. **Image Segmentation**: Large images split into overlapping patches
3. **Batch Processing**: Process patches according to `--bs` parameter
4. **Result Fusion**: Combine patches with overlap handling
5. **Output**: Saves to `load_weights_path/../Inference/`

### Key Parameters by Modality

#### 2D Wide-field:
```bash
--iterations 50000
--batch_size 4
--start_lr 5e-5
--input_x 512 --input_y 512
--insert_xy 8
--denoise_loss_weight 0.5
--TV_rate 0.01
```

#### 3D Wide-field/Confocal:
```bash
--iterations 10000
--batch_size 1
--start_lr 1e-4
--input_x 128 --input_y 128 --input_z 32
--insert_xy 8 --insert_z 2
--TV_rate 0.1 --Hess_rate 0.02
```

#### SIM (2D/3D):
```bash
--augment_flag True
--TV_rate 0.1
--model_name 'RCAN3D_SIM' # for 3D SIM
```

### Monitoring and Output

#### Training Monitoring:
```bash
tensorboard --logdir <save_weights_dir>/<save_weights_name>/graph
```

#### Output Structure:
```
saved_models/
├── [experiment_name]/
│   ├── saved_model/        # Model weights (.h5)
│   ├── Inference/          # Inference results
│   ├── graph/              # TensorBoard logs
│   └── test_data/          # Test images
```

### Memory and Performance Tips

- **Large Images**: Use `num_seg_window_x/y` parameters for segmentation
- **Overlap**: Set `overlap_x/y` >= 16 pixels to avoid artifacts
- **3D Processing**: Reduce `batch_size` to 1-2 for memory constraints
- **GPU Memory**: Use `--gpu_memory_fraction` to limit usage
- **Mixed Precision**: Enable with `--mixed_precision_training True`

# PyTorch Lightning Pure Deconvolution Implementation

## Current Task (2025-01-24)
**User Requirements**: Convert TensorFlow version to PyTorch Lightning, keeping **only deconvolution** (remove denoising stage).

### Working Environment
- **Working Directory**: `/home/aero/charliechang/projects/ZS-DeconvNet/`
- **Conda Environment**: `zs-deconvnet_pytorch`
- **Code Location**: `PyTorch_Deconv/`

### Data Structure
```
PyTorch_Deconv/Data/
├── Train/                  # Training data (3D TIFF stacks)
│   ├── DPM4Xsample_2070.tif
│   └── THXsample_2070.tif
├── PSF/                    # Point Spread Function
│   └── psf_emLambda525_dxy0.0313_NA1.3.tif
├── Inference/              # Inference input
└── InferenceResult/        # Output results
```

### Training Logic (Self-Supervised Deconvolution)
```
Blurred Image (input) → U-Net → Clear Image (prediction) → PSF Convolution → Re-blurred Image → Compare with Original Blurred Image (target)
```
- **Key Insight**: input = target = original blurred image (self-supervised)
- **Learning Goal**: Model learns blur → clear inverse mapping
- **No paired data needed**: Only original blurred microscopy images

### Inference Logic
```
3D TIFF Blurred Image → Slice-by-slice Processing → 2x Super-resolution Clear Image → Save as *_deconvolved.tif
```

### Current Issues to Fix
1. **PSF Loss Cropping Bug**: Incorrect tensor cropping causing all-black outputs
2. **Output Size Handling**: Ensure proper 2x super-resolution output
3. **Configuration**: Set max_epochs=100 for initial testing

### Critical Implementation Details
- **Pure Deconvolution**: Based on TensorFlow version's second stage only
- **Self-Supervised**: input=target is CORRECT for deconvolution learning
- **PSF-Constrained**: Physics-based learning using known microscope PSF
- **2x Super-Resolution**: Simultaneously deblur and upscale