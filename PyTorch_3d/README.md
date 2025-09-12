# ZS-DeconvNet PyTorch 3D Implementation

PyTorch Lightning implementation of ZS-DeconvNet for 3D fluorescence microscopy denoising and deconvolution.

## Project Structure

```
PyTorch_3d/
├── models/                    # Neural network models
│   ├── twostage_RCAN3D.py    # RCAN3D two-stage model
│   ├── twostage_Unet3D.py    # UNet3D two-stage model
│   └── blocks.py             # Shared building blocks
├── utils/                     # Utility functions
│   ├── utils.py              # General utilities
│   ├── loss.py               # Loss functions and PSF utilities
│   └── data_loader.py        # PyTorch Dataset and DataLoader
├── config/
│   └── train_3d.yaml         # Training configuration
├── Train_ZSDeconvNet_3D.py   # Main training script
├── Infer_3D.py               # Inference script
└── requirements.txt          # Dependencies
```

## Installation

1. Create and activate conda environment:
```bash
conda create -n zs-deconvnet_pytorch python=3.9
conda activate zs-deconvnet_pytorch
```

2. Install dependencies:
```bash
cd PyTorch_3d
pip install -r requirements.txt
```

## Usage

### Training

```bash
python Train_ZSDeconvNet_3D.py \
    --model twostage_RCAN3D \
    --psf_path /path/to/psf.tif \
    --data_dir /path/to/training/data \
    --save_weights_dir ./outputs \
    --upsample_flag 0 \
    --max_epochs 100 \
    --batch_size 3
```

**Required arguments:**
- `--psf_path`: Path to PSF TIFF file
- `--data_dir`: Path to training data directory (should contain `input/` and `gt/` subdirectories)

**Key optional arguments:**
- `--model`: Model type (`twostage_RCAN3D` or `twostage_Unet3D`)
- `--upsample_flag`: Enable 2x super-resolution (0 or 1)
- `--save_weights_dir`: Output directory for trained models
- `--max_epochs`: Number of training epochs
- `--batch_size`: Training batch size

### Inference

```bash
python Infer_3D.py \
    --load_weights_path /path/to/trained/model.ckpt \
    --input_dir /path/to/input/images \
    --output_dir /path/to/output \
    --model twostage_RCAN3D \
    --upsample_flag 0
```

**For large images, use tiling:**
```bash
python Infer_3D.py \
    --load_weights_path /path/to/model.ckpt \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --use_tiling \
    --tile_size_z 64 --tile_size_y 256 --tile_size_x 256 \
    --overlap_z 8 --overlap_y 32 --overlap_x 32
```

## Data Format

Training data should be organized as:
```
training_data/
├── input/     # Noisy input images (TIFF format)
└── gt/        # Ground truth images (TIFF format)
```

- Images should be 3D TIFF stacks (Z, Y, X)
- Corresponding input and GT files should have the same name
- Typical patch size: 64×64×13 (configurable)

## Model Architecture

### Two-Stage Design
1. **Stage 1 (Denoising)**: Removes noise while preserving structures
2. **Stage 2 (Deconvolution)**: Performs deconvolution with optional 2x super-resolution

### Supported Models
- **RCAN3D**: Residual Channel Attention Network with 3D operations
- **UNet3D**: 3D U-Net encoder-decoder architecture

### Loss Functions
- **NBR2NBR Loss**: Self-supervised noise-to-noise training
- **PSF Loss**: Physics-informed loss with PSF convolution
- **Regularization**: Total Variation and Hessian penalties

## Key Features

- **Self-supervised learning**: No need for paired clean data
- **Physics-informed training**: PSF-based convolution loss
- **3D processing**: Native 3D convolution operations
- **Super-resolution**: Optional 2x upsampling
- **Tiling support**: Process large images efficiently
- **PyTorch Lightning**: Modern training framework
- **Mixed precision**: 16-bit training support

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- CUDA-capable GPU (recommended)

See `requirements.txt` for complete dependency list.

## Notes

This implementation is fully compatible with the original TensorFlow version while leveraging modern PyTorch features and best practices.