#!/bin/bash
#
# Custom 3D training script following ReadMe.md guide
# Using our generated augmented data and PSF
#

cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

echo "Starting ZS-DeconvNet 3D training..."

python Train_ZSDeconvNet_3D.py \
  --psf_path '../../Python_PSF/PSFoutput/optical/PSF_optical_NA1.1_lambda525_size79_Z23.tif' \
  --data_dir 'data/augmented_data/iUExM/' \
  --folder 'roiC_crop128_1128_0901_1006' \
  --test_images_path 'data/ori_input/iUExM/roiC_crop128_1128.tif' \
  --save_weights_dir './my_models/iUExM' \
  --save_weights_suffix '_PSF_optical_NA1.1_lambda525_size79_Z23_upsample0' \
  --model 'twostage_RCAN3D' \
  --upsample_flag 0 \
  --iterations 5000 \
  --test_interval 500 \
  --valid_interval 500 \
  --batch_size 3 \
  --gpu_id 1 \
  --gpu_memory_fraction 0.8 \
  --mixed_precision_training 1 \
  --start_lr 1e-4 \
  --input_y 64 \
  --input_x 64 \
  --input_z 13 \
  --insert_xy 8 \
  --insert_z 2 \
  --dx 0.0926 \
  --dz 0.3704 \
  --dxpsf 0.0926 \
  --dzpsf 0.05 \
  --background 100 \
  --mse_flag 0 \
  --TV_weight 0.0 \
  --Hess_weight 0.1

echo "Training completed!"