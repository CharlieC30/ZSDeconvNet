#!/bin/bash
#
# ZS-DeconvNet 3D Training Demo Script - PyTorch Version
# Equivalent to original TensorFlow train_demo_3D.sh
#

# ------------------------------- Training Arguments -------------------------------

# Models
# --model: "twostage_RCAN3D" or "twostage_Unet3D"
# --upsample_flag: 0 or 1, whether the network upsamples the image

# Training settings
# --load_all_data: 1 for load all training set into memory before training for faster computation, 0 for not
# --gpu_id: the gpu device you want to use in current task
# --iterations: total training iterations
# --test_interval: iteration interval of testing and model saving
# --valid_interval: iteration interval of validation
# --batch_size: batch size for training

# Learning rate
# --start_lr: initial learning rate of training, typically set as 1e-4
# --lr_decay_factor: learning rate decay factor, typically set as 0.5

# Data settings
# You need to arrange your data in the format: data_dir+folder+'/input/' and data_dir+folder+'/gt/'
# --psf_path: path of corresponding PSF. Supports .tif format
# --data_dir: the root directory of training data folder
# --folder: the name of training data folder
# --test_images_path: the root path of test data (not implemented in PyTorch version yet)
# --save_weights_dir: root directory where model weights will be saved
# --background: set to the value you want to extract in images

# Image patch settings
# --input_y: the height of input image patches
# --input_x: the width of input image patches
# --insert_xy: padded blank margin in pixels
# --input_z: the depth of input image patches
# --insert_z: padded blank margin in axial direction (how many slices)
# --dx: sampling interval in x direction (um) for training data
# --dz: sampling interval in z direction (um) for training data
# --dxpsf: sampling interval in x direction (um) of raw PSF
# --dzpsf: sampling interval in z direction (um) of raw PSF
# --norm_flag: 1 for minmax normalization, 0 for /65535, 2 for /max

# Loss functions
# --mse_flag: 0 for MAE, 1 for MSE
# --TV_weight: the weighting factor for TV regularization term
# --Hess_weight: the weighting factor for Hessian regularization term

# PyTorch specific
# --device: 'auto', 'cpu', or 'cuda' for device selection

cd /home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_3d

# Activate PyTorch environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet_pytorch

# ------------------------------- Examples -------------------------------

echo "Starting ZS-DeconvNet 3D Training (PyTorch Version)..."
echo "======================================================"

# Example 1: Basic training with RCAN3D
python Train_ZSDeconvNet_3D.py \
    --psf_path 'path/to/your/PSF.tif' \
    --data_dir 'path/to/your/training/data/' \
    --folder 'training_folder_name' \
    --model 'twostage_RCAN3D' \
    --upsample_flag 0 \
    --iterations 10000 \
    --test_interval 1000 \
    --valid_interval 1000 \
    --batch_size 3 \
    --gpu_id 0 \
    --start_lr 1e-4 \
    --lr_decay_factor 0.5 \
    --input_y 64 \
    --input_x 64 \
    --input_z 13 \
    --insert_xy 8 \
    --insert_z 2 \
    --dx 0.0926 \
    --dz 0.3704 \
    --dxpsf 0.0926 \
    --dzpsf 0.05 \
    --norm_flag 0 \
    --background 100 \
    --mse_flag 0 \
    --TV_weight 0.0 \
    --Hess_weight 0.1 \
    --save_weights_dir './models'

echo "Training completed! Check ./models for saved models."