# PyTorch Lightning Deconvolution Network

A modern PyTorch Lightning implementation of ZS-DeconvNet for single-stage 2D deconvolution. This implementation processes each z-slice of 3D TIFF files as independent 2D images for deconvolution.

## Features

- **Single-stage deconvolution**: Removed the denoising stage, focuses only on deconvolution
- **PyTorch Lightning**: Modern training framework with automatic logging, checkpointing, and multi-GPU support
- **3D TIFF support**: Processes each z-slice independently as 2D images
- **PSF-based loss**: Implements PSF convolution loss with regularization terms
- **Flexible data handling**: Supports various input formats and patch-based training
- **Tiled inference**: Supports processing of large images through tiling

## Installation

1. Create a conda environment:
```bash
conda create -n pytorch-deconv python=3.9
conda activate pytorch-deconv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your data as follows:
```
Data/
├── Train/           # Training 3D TIFF files
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── Inference/       # Inference 3D TIFF files
│   ├── test1.tif
│   ├── test2.tif
│   └── ...
└── PSF/            # Point Spread Function files
    └── psf_emLambda525_dxy0.0313_NA1.3.tif
```

## Usage

### Training

1. **Prepare your configuration**: Edit `config.yaml` to match your setup:
   - Adjust batch size, learning rate, and other hyperparameters
   - Set the correct PSF sampling intervals
   - Configure regularization weights

2. **Run training**:
```bash
python train.py \
    --config config.yaml \
    --data_dir ./Data \
    --psf_path ./Data/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
    --output_dir ./outputs \
    --gpus 1
```

3. **Monitor training**: View logs in TensorBoard:
```bash
tensorboard --logdir ./outputs/deconv_logs
```

### Inference

Run inference on new images:
```bash
python infer.py \
    --checkpoint ./outputs/deconv_logs/version_0/checkpoints/best.ckpt \
    --input_dir ./Data/Inference \
    --output_dir ./results \
    --device cuda
```

For large images, use tiling:
```bash
python infer.py \
    --checkpoint ./outputs/deconv_logs/version_0/checkpoints/best.ckpt \
    --input_dir ./Data/Inference \
    --output_dir ./results \
    --tile_size 512 \
    --overlap 64 \
    --device cuda
```

## Configuration

Key configuration parameters in `config.yaml`:

### Model Parameters
- `conv_block_num`: Number of U-Net blocks (default: 4)
- `upsample_flag`: Whether to upsample output (default: true)
- `insert_xy`: Padding size for network input (default: 16)

### Loss Parameters
- `hessian_weight`: Main regularization term (default: 0.02)
- `tv_weight`: Total variation regularization (default: 0.0)
- `use_mse`: Use MSE loss instead of MAE (default: false)

### Training Parameters
- `batch_size`: Training batch size (default: 4)
- `patch_size`: Size of training patches (default: 128)
- `lr`: Learning rate (default: 5e-5)
- `max_epochs`: Maximum training epochs (default: 250)

### PSF Parameters
- `target_dx/dy`: Target sampling intervals in micrometers
- `psf_dx/dy`: PSF sampling intervals (null = auto-detect)

## Architecture

The implementation consists of:

1. **DeconvUNet** (`src/models/deconv_unet.py`): Single-stage U-Net for deconvolution
2. **PSFConvolutionLoss** (`src/losses/psf_loss.py`): PSF-based loss with regularization
3. **DeconvDataModule** (`src/data/datamodule.py`): Data loading for 3D TIFF files
4. **DeconvolutionLightningModule** (`src/models/lightning_module.py`): Training logic
5. **PSFProcessor** (`src/utils/psf_utils.py`): PSF loading and processing

## Key Differences from Original

1. **Removed denoising stage**: Only the second-stage deconvolution network is implemented
2. **3D TIFF handling**: Each z-slice is processed independently as a 2D image
3. **Modern framework**: Uses PyTorch Lightning for training
4. **Flexible data format**: Supports various input configurations

## Advanced Usage

### Custom Loss Functions

You can modify the loss function configuration:
```yaml
loss:
  tv_weight: 0.01      # Add total variation regularization
  hessian_weight: 0.02 # Hessian regularization (recommended)
  l1_weight: 0.001     # L1 regularization on output
```

### Multi-GPU Training

For multi-GPU training:
```bash
python train.py \
    --config config.yaml \
    --data_dir ./Data \
    --psf_path ./Data/PSF/psf.tif \
    --gpus 2
```

### Mixed Precision Training

Enable mixed precision for faster training:
```yaml
trainer:
  precision: 16  # Use 16-bit precision
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or patch size
2. **PSF loading errors**: Check PSF file format and sampling intervals
3. **Training instability**: Reduce learning rate or increase regularization

### Performance Tips

1. Use SSD storage for faster data loading
2. Increase `num_workers` for better data loading performance
3. Use larger batch sizes if memory allows
4. Enable mixed precision training for speed

## Output Format

- Training outputs: Checkpoints and logs in `outputs/`
- Inference outputs: Deconvolved images as `*_deconvolved.tif`
- All outputs are saved as 16-bit TIFF files

## Citation

This implementation is based on the ZS-DeconvNet paper. If you use this code, please cite the original work.