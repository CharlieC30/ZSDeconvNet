# ZS-DeconvNet PyTorch Implementation

A PyTorch Lightning implementation of Zero-Shot Deconvolution Network for 2D fluorescence microscopy image deconvolution.

## Overview

This implementation provides a single-stage 2D U-Net for deconvolution with PSF-based loss function. It processes 3D TIFF files by treating each z-slice as an independent 2D image for deconvolution.

## Features

- Single-stage U-Net architecture (deconvolution only, denoising stage removed)
- PSF convolution loss with Total Variation and Hessian regularization
- 3D TIFF processing with z-slice independence
- PyTorch Lightning framework for modern training
- Automatic experiment organization with timestamps
- GPU acceleration support

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda or pip package manager

### Setup Environment

1. Create and activate conda environment:
```bash
conda create -n zs-deconvnet_pytorch python=3.9
conda activate zs-deconvnet_pytorch
```

2. Install dependencies:
```bash
cd PyTorch_Deconv
pip install -r requirements.txt
```

## Directory Structure

```
PyTorch_Deconv/
├── src/                    # Core modules
│   ├── models/            # U-Net model and Lightning module
│   ├── losses/            # PSF convolution loss functions
│   ├── data/              # Data loading and processing
│   └── utils/             # PSF utilities
├── experiments/           # Training experiments (auto-generated)
│   └── MMDD_HHMM/        # Timestamped experiment folders
├── Data/                  # Data directory
│   ├── Train/            # Training TIFF files
│   ├── Inference/        # Input files for inference
│   └── PSF/              # PSF files
├── config.yaml           # Training configuration
├── train.py              # Training script
├── infer.py              # Inference script
└── requirements.txt      # Dependencies
```

## Usage

### Training

Train a new model:

```bash
python train.py \
    --config config.yaml \
    --data_dir Data \
    --psf_path Data/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
    --gpus 1
```

Resume training from checkpoint:

```bash
python train.py \
    --config config.yaml \
    --data_dir Data \
    --psf_path Data/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
    --resume experiments/1219_1430/checkpoints/last.ckpt \
    --gpus 1
```

### Inference

Run inference on TIFF files:

```bash
python infer.py \
    --checkpoint experiments/1219_1430/final_model.ckpt \
    --input_dir Data/Inference \
    --output_dir Data/InferenceResult \
    --device cuda
```

## Configuration

Key parameters in `config.yaml`:

### Model Configuration
- `input_channels`: Input image channels (default: 1)
- `output_channels`: Output image channels (default: 1)
- `conv_block_num`: Number of encoder/decoder blocks (default: 4)
- `conv_num`: Convolutions per block (default: 3)
- `upsample_flag`: Enable 2x super-resolution (default: true)
- `insert_xy`: Padding size (default: 16)

### Loss Function
- `tv_weight`: Total Variation regularization (default: 0.0)
- `hessian_weight`: Hessian regularization (default: 0.02)
- `l1_weight`: L1 regularization (default: 0.0)
- `use_mse`: Use MSE instead of MAE (default: false)

### Training Parameters
- `lr`: Learning rate (default: 0.00005)
- `max_epochs`: Maximum training epochs (default: 250)
- `batch_size`: Training batch size (default: 4)
- `patch_size`: Image patch size (default: 128)

### PSF Configuration
- `target_dx`: Target sampling interval x (default: 0.0313)
- `target_dy`: Target sampling interval y (default: 0.0313)

## Data Format

### Training Data
Place TIFF files in `Data/Train/`. Both 2D and 3D TIFF files are supported.

### PSF Files
PSF files should be in TIFF format and placed in `Data/PSF/`. The PSF will be automatically normalized and cropped for optimal computation.

### Output Format
- Input: 512x512 (or any size)
- Output: 1024x1024 (2x super-resolution when `upsample_flag=true`)
- 3D files: Each z-slice processed independently

## Experiment Management

Each training run creates a timestamped experiment folder:
- Format: `experiments/MMDD_HHMM/` (e.g., `experiments/1219_1430/`)
- Contains: checkpoints, logs, configuration, and final model
- Prevents accidental overwriting of previous experiments

## Hardware Requirements

### Minimum
- GPU: 4GB VRAM
- RAM: 8GB
- Storage: 10GB free space

### Recommended
- GPU: 8GB+ VRAM (RTX 3070 or better)
- RAM: 16GB+
- Storage: 50GB+ for multiple experiments

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in config.yaml
2. **No TIFF files found**: Check file paths and extensions (.tif/.tiff)
3. **PSF loading errors**: Ensure PSF file is valid TIFF format
4. **Import errors**: Verify all dependencies are installed

### Performance Tips

1. Use `--gpus 1` for GPU acceleration
2. Adjust `num_workers` based on CPU cores
3. Use SSD storage for faster data loading
4. Monitor GPU memory usage during training

## Technical Details

### Architecture
- Based on original ZS-DeconvNet paper implementation
- Modified to single-stage deconvolution (no denoising)
- U-Net with skip connections and channel reduction in decoder
- Final 2x upsampling for super-resolution

### Loss Function
- PSF convolution loss for physics-based deconvolution
- Hessian regularization for smoothness
- Optional Total Variation and L1 regularization

### Data Processing
- Percentile normalization (0-100th percentile to [0,1])
- Random patch extraction during training
- Data augmentation: rotation and flipping
- Padding handling for network input/output