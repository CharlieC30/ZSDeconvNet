# PyTorch Lightning Pure Deconvolution Network

A clean PyTorch Lightning implementation of ZS-DeconvNet focused exclusively on **deconvolution**. This implementation removes the denoising stage and processes each z-slice of 3D TIFF files as independent 2D images for pure deconvolution with 2x super-resolution.

## Features

- **Pure deconvolution**: Only the second-stage deconvolution network (removed denoising stage)
- **Self-supervised learning**: No paired data required - uses input=target strategy
- **2x super-resolution**: Simultaneous deblurring and resolution enhancement
- **PyTorch Lightning**: Modern training framework with automatic logging and checkpointing
- **3D TIFF support**: Processes each z-slice independently as 2D images
- **PSF-based physics loss**: Implements PSF convolution loss with regularization terms
- **Date-time naming**: Automatic MMDD_HHMM output naming for easy management
- **Tiled inference**: Supports processing of large images through tiling

## Installation

1. Create and activate conda environment:
```bash
conda create -n zs-deconvnet_pytorch python=3.9
conda activate zs-deconvnet_pytorch
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Structure

Organize your data as follows:
```
Data/
├── Train/              # Training 3D TIFF files (for self-supervised learning)
│   ├── DPM4Xsample_2070.tif
│   ├── THXsample_2070.tif
│   └── ... (blurred microscopy images)
├── InferenceInput/     # Inference input 3D TIFF files
│   ├── test1.tif
│   ├── test2.tif
│   └── ... (images to be deconvolved)
├── InferenceResult/    # Output deconvolved images (auto-created)
│   ├── test1_deconvolved.tif
│   └── test2_deconvolved.tif
├── Output/            # Training outputs with date-time naming
│   ├── deconv_logs/
│   │   ├── 0124_1430/  # MMDD_HHMM format
│   │   └── 0124_1545/
│   └── final_model.ckpt
└── PSF/               # Point Spread Function files
    └── psf_emLambda525_dxy0.0313_NA1.3.tif
```

## Self-Supervised Learning Principle

This implementation uses **input = target** strategy for pure deconvolution:
```
Blurred Image (input) → U-Net → Clear Image (prediction) → PSF Convolution → Re-blurred Image → Compare with Original Blurred Image (target)
```

**Key insight**: The model learns to output a clear image that, when convolved with the PSF, reproduces the original blurred input. This requires no paired ground truth data!

## Usage

### Training

1. **Prepare your configuration**: The default `config.yaml` is optimized for pure deconvolution:
   - `max_epochs: 100` (for quick testing, increase for production)
   - `batch_size: 4` (adjust based on GPU memory)
   - `hessian_weight: 0.02` (main regularization for edge preservation)
   - `learning_rate: 5e-5` (optimized for stable convergence)

2. **Run training**:
```bash
python train.py \
    --config config.yaml \
    --data_dir Data \
    --psf_path Data/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
    --output_dir Data/Output \
    --gpus 1
```

**Expected output**: Loss decreases from ~0.13 to ~0.009 over 100 epochs. Final validation loss around 0.010.

3. **Monitor training**: View logs in TensorBoard:
```bash
tensorboard --logdir Data/Output/deconv_logs
```

### Inference

Run inference on new images:
```bash
python infer.py \
    --checkpoint Data/Output/final_model.ckpt \
    --input_dir Data/InferenceInput \
    --output_dir Data/InferenceResult \
    --device cuda
```

**Expected output**: 
```
Processing: Data/InferenceInput/sample.tif
Image shape: (50, 512, 512)
Processing slices: 100%|█████| 50/50 [00:03<00:00, 15.77it/s]
Result shape: (50, 1024, 1024), range: [0.0425, 1.0521]
Saved to: Data/InferenceResult/sample_deconvolved.tif
```

For large images, use tiling:
```bash
python infer.py \
    --checkpoint Data/Output/final_model.ckpt \
    --input_dir Data/InferenceInput \
    --output_dir Data/InferenceResult \
    --tile_size 512 \
    --overlap 64 \
    --device cuda
```

## Configuration

Key configuration parameters in `config.yaml`:

### Model Parameters
- `conv_block_num: 4`: Number of U-Net encoder/decoder blocks
- `upsample_flag: true`: Enable 2x super-resolution output
- `insert_xy: 16`: Input padding size (prevents edge artifacts during convolution)

### Loss Parameters (Physics-based)
- `hessian_weight: 0.02`: **Critical parameter** - Hessian regularization for edge preservation
- `tv_weight: 0.0`: Total variation regularization (optional)
- `l1_weight: 0.0`: L1 sparsity regularization (optional)
- `use_mse: false`: Use MAE loss instead of MSE (more stable)

### Training Parameters
- `batch_size: 4`: Training batch size (reduce to 2 if GPU memory < 8GB)
- `patch_size: 128`: Training patch size extracted from images
- `lr: 5e-5`: Learning rate (stable convergence rate)
- `max_epochs: 100`: Training epochs (sufficient for convergence)

### PSF Parameters
- `target_dx: 0.0313`: Target image sampling interval (micrometers)
- `target_dy: 0.0313`: Target image sampling interval (micrometers)
- `psf_dx/dy: null`: PSF sampling intervals (auto-detected from file)

### Date-Time Output Naming
- Format: `MMDD_HHMM` (e.g., `0124_1535` for Jan 24, 15:35)
- Conflicts resolved by adding seconds: `MMDD_HHMM_SS`
- Logs saved to: `Data/Output/deconv_logs/MMDD_HHMM/`

## Architecture

The clean implementation consists of:

1. **DeconvUNet** (`src/models/deconv_unet.py`): Single-stage U-Net for pure deconvolution
2. **PSFConvolutionLoss** (`src/losses/psf_loss.py`): Physics-based PSF loss with Hessian regularization
3. **DeconvDataModule** (`src/data/datamodule.py`): Streamlined data loading for 3D TIFF files
4. **DeconvolutionLightningModule** (`src/models/lightning_module.py`): Training logic with self-supervised loss
5. **PSFProcessor** (`src/utils/psf_utils.py`): PSF loading and cropping utilities

## Key Differences from Original TensorFlow Version

1. **Pure deconvolution**: Removed the first-stage denoising network entirely
2. **Self-supervised strategy**: Uses input=target with PSF-constrained learning
3. **Modern PyTorch Lightning**: Replaces TensorFlow 1.x with modern framework
4. **Slice-based processing**: Each z-slice processed independently as 2D
5. **Date-time naming**: Replaced version numbers with intuitive MMDD_HHMM format
6. **Cleaned codebase**: Removed all debug/test files for production use
7. **Fixed PSF loss**: Corrected tensor cropping logic that caused black outputs

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

1. **Model loading warnings**: The warning about 'loss_fn.psf_loss.psf' keys can be safely ignored
   ```
   UserWarning: Found keys that are not in the model state dict...
   ```

2. **CUDA out of memory**: Reduce batch size
   ```yaml
   data:
     batch_size: 2  # or batch_size: 1
   ```

3. **Training loss not decreasing**: 
   - Ensure `hessian_weight: 0.02` is set (critical for convergence)
   - Check PSF file is valid .tif format
   - Verify training data contains multiple images

4. **Black or zero outputs**: This issue was fixed in the current version
   - Check output files: `tifffile.imread('result.tif').max() > 0`

5. **Tensor Cores warning**: Can be safely ignored or add to training script:
   ```python
   torch.set_float32_matmul_precision('medium')
   ```

### Performance Tips

1. **GPU memory**: Adjust batch_size based on available VRAM (4GB = batch_size 2, 8GB+ = batch_size 4)
2. **Data loading**: Increase `num_workers` if you have multiple CPU cores
3. **Mixed precision**: Add `precision: 16` to trainer config for faster training
4. **Monitor progress**: `tensorboard --logdir Data/Output/deconv_logs`

### Fixed Issues in This Version

- **Black output problem**: Fixed PSF loss tensor cropping logic
- **Model architecture**: Removed incorrect final ReLU activation
- **Training stability**: Optimized loss function weights
- **Clean output**: Removed excessive debug logging

## Output Format

- **Training**: Checkpoints in `Data/Output/deconv_logs/MMDD_HHMM/checkpoints/`
- **Final model**: `Data/Output/final_model.ckpt`
- **Inference**: Deconvolved images as `*_deconvolved.tif` in specified output directory
- **Format**: All outputs are 16-bit TIFF files with percentile normalization

## Citation

This implementation is based on the ZS-DeconvNet paper. If you use this code, please cite the original work.