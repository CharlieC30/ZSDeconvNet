# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ZS-DeconvNet is a zero-shot learning deep neural network for instant denoising and super-resolution in optical fluorescence microscopy. The current working directory is `Python_MATLAB_Codes/` which contains the main TensorFlow 2.5.0 implementation.

### Directory Structure
- `train_inference_python/`: Main Python implementation with training and inference scripts
- `data_augment_recorrupt_matlab/`: MATLAB codes for training data generation and PSF simulation
- `saved_models/`: Pre-trained models for different microscopy modalities
- `your_augmented_datasets/`: Output directory for generated training datasets

## Development Commands

### Python Environment Setup
```bash
# Create conda environment
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet

# Install dependencies (from train_inference_python directory)
cd train_inference_python
pip install -r requirements.txt

# Install CUDA (if using GPU)
conda install cudatoolkit==11.3.1
conda install cudnn==8.2.1
```

### Training Commands
All training scripts must be run from `train_inference_python/` directory:

```bash
cd train_inference_python

# 2D model training
python Train_ZSDeconvNet_2D.py --otf_or_psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME] --test_images_path [TEST_PATH]

# 3D model training
python Train_ZSDeconvNet_3D.py --psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME] --test_images_path [TEST_PATH]

# Use demo scripts (edit paths first):
./train_demo_2D.sh        # 2D wide-field data
./train_demo_3D.sh        # 3D wide-field, confocal, LLSM data
./train_demo_2DSIM.sh     # 2D reconstructed SIM data
./train_demo_3DSIM.sh     # 3D reconstructed SIM data
```

### Inference Commands
```bash
cd train_inference_python

# 2D inference
python Infer_2D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# 3D inference
python Infer_3D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# Use demo scripts (edit paths first):
./infer_demo_2D.sh
./infer_demo_3D.sh
```

### Data Augmentation (Python)
```bash
cd train_inference_python

# Generate augmented 3D training data
python DataAugmFor3d_python.py --input_dir [RAW_DATA_DIR] --output_dir [OUTPUT_DIR]
```

## Architecture Overview

### Core Components

1. **Two-Stage Architecture**: All models use a two-stage approach:
   - Stage 1: Denoising network
   - Stage 2: Deconvolution network with PSF-based loss

2. **Model Types**:
   - `twostage_Unet`: 2D U-Net based model
   - `twostage_Unet3D`: 3D U-Net based model
   - `twostage_RCAN3D`: 3D Residual Channel Attention Network

3. **Key Files** (in `train_inference_python/`):
   - `models/twostage_Unet.py`: 2D U-Net architecture
   - `models/twostage_Unet3D.py`: 3D U-Net architecture
   - `models/twostage_RCAN3D.py`: 3D Residual Channel Attention Network
   - `utils/data_loader.py`: Data loading and preprocessing functions
   - `utils/loss.py`: Loss functions including frequency domain losses
   - `utils/utils.py`: General utilities for PSF handling, normalization
   - `utils/augment_sim_img.py`: Data augmentation for SIM images
   - Training scripts: `Train_ZSDeconvNet_2D.py`, `Train_ZSDeconvNet_3D.py`, `Train_ZSDeconvNet_2DSIM.py`, `Train_ZSDeconvNet_3DSIM.py`
   - Inference scripts: `Infer_2D.py`, `Infer_3D.py`
   - `DataAugmFor3d_python.py`: Python-based data augmentation for 3D datasets

### Data Structure
Training data should be organized as:
```
data_dir/
├── folder_name/
│   ├── input/          # Noisy input images
│   └── gt/             # Ground truth images (for supervised training)
```

For zero-shot training, only input images are needed with data augmentation.

### Loss Functions
- **Denoising Loss**: MSE or MAE between denoised and target
- **Deconvolution Loss**: Frequency domain loss using PSF/OTF
- **Regularization**: Hessian, TV, L1 regularization terms

### PSF/OTF Handling
- Supports both PSF (.tif) and OTF (.mrc) formats
- Automatic interpolation for pixel size matching
- PSF normalization before loss calculation

## Key Parameters

### Training Parameters
- `--gpu_id`: GPU device ID
- `--iterations`: Total training iterations
- `--start_lr`: Initial learning rate (5e-5 for 2D, 1e-4 for 3D)
- `--batch_size`: Batch size (4 for 2D, 2-3 for 3D)
- `--input_x/y/z`: Patch dimensions (128 for 2D, 64 for 3D)
- `--dx/dy/dz`: Pixel spacing in micrometers
- `--upsample_flag`: Whether to perform super-resolution

### Loss Weights
- `--denoise_loss_weight`: Weight for denoising loss
- `--Hess_rate`: Hessian regularization (0.02 for 2D, 0.1 for 3D)
- `--TV_rate`: Total variation regularization
- `--l1_rate`: L1 regularization

### Data Augmentation (Zero-shot)
- `--alpha`: Noise magnification factor [1-2]
- `--beta1`: Poisson noise factor [0.5-1.5]
- `--beta2`: Gaussian noise variance (estimated from data)

## Development Notes

### GPU Memory Management
- Set `--gpu_memory_fraction` to limit GPU usage
- Use `--mixed_precision_training` for memory efficiency
- Adjust `--batch_size` based on available memory

### Model Saving
- Models saved in `saved_models/` (root level) or `my_models_3d/` (in train_inference_python/)
- Weights saved as `.h5` files at specified intervals
- Configuration saved as `config.txt`
- Logs saved to `graph/` subdirectory for TensorBoard monitoring

### Testing and Monitoring
- Use `--test_interval` to specify validation frequency during training
- Test images specified via `--test_images_path`
- Inference results saved to `Inference/` subdirectory within model folder
- Monitor training with: `tensorboard --logdir [model_dir]/graph`

### Dependencies and Requirements
- TensorFlow 2.5.0 with GPU support
- Required packages: `imageio`, `tifffile`, `scipy==1.7.1`, `opencv-python`
- CUDA 11.3/11.4 and cuDNN 8.2 for GPU acceleration

## Common Issues

1. **CUDA compatibility**: Ensure TensorFlow 2.5.0 matches CUDA 11.3-11.4/cuDNN 8.2
2. **Memory errors**: Reduce `--batch_size` or patch dimensions (`--input_x/y/z`)
3. **PSF format**: Verify PSF dimensions are odd numbers and properly normalized
4. **Working directory**: Always run training/inference from `train_inference_python/` directory
5. **Demo scripts**: Edit absolute paths in `.sh` files before running
6. **Data organization**: Ensure training data follows `data_dir/folder/input/` and `data_dir/folder/gt/` structure

## Quick Start Workflow

1. **Setup environment**: Create conda environment and install dependencies
2. **Download data**: Get demo datasets from Google Drive and place in `saved_models/`
3. **Test inference**: Run `./infer_demo_2D.sh` or `./infer_demo_3D.sh` (edit paths first)
4. **Train new model**: Prepare data, edit demo training script, run training
5. **Monitor training**: Use TensorBoard to track progress