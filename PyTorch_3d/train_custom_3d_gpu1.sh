#!/bin/bash
#
# Custom 3D Training Script - PyTorch Version
# Based on your existing TensorFlow train_custom_3d_gpu1.sh
# Using your generated data and PSF
#

# Switch to correct directory
cd /home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_3d

# Ensure using PyTorch environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet_pytorch

echo "Starting ZS-DeconvNet 3D Training (PyTorch Version)..."
echo "Using Python-generated augmented data"
echo "========================================"

# Execute training with your custom parameters
python Train_ZSDeconvNet_3D.py \
  --psf_path '../../Python_PSF/PSFoutput/other/PSF_XY1.88um_Z15.04um_oddZ_111.tif' \
  --data_dir '../your_augmented_datasets/aisr/' \
  --folder 'aisr122424_roi_0905_1200_100' \
  --save_weights_dir './my_models_3d' \
  --model 'twostage_RCAN3D' \
  --upsample_flag 0 \
  --iterations 500 \
  --test_interval 500 \
  --valid_interval 100 \
  --batch_size 2 \
  --gpu_id 1 \
  --start_lr 1e-4 \
  --lr_decay_factor 0.5 \
  --input_y 64 \
  --input_x 64 \
  --input_z 13 \
  --insert_xy 8 \
  --insert_z 2 \
  --dx 0.5 \
  --dz 2 \
  --dxpsf 1 \
  --dzpsf 1 \
  --background 100 \
  --mse_flag 0 \
  --TV_weight 0.0 \
  --Hess_weight 0.1 \
  --norm_flag 0 \
  --load_all_data 1

echo "Training completed!"
echo "Model weights saved in: ./my_models_3d/"
echo "Use the saved .pth files for inference."