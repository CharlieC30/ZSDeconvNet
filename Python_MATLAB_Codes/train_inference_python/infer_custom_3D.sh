#!/bin/bash
#
# Custom inference script using trained ZS-DeconvNet model
# Following ReadMe.md section 5 guide
#

cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

echo "Starting ZS-DeconvNet 3D inference..."

# export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

python Infer_3D.py \
  --input_dir 'data/ori_input/iUExM/roiC_crop128_1128.tif' \
  --load_weights_path './my_models/iUExM/roiC_crop128_1128_0901_1006_twostage_RCAN3D_PSF_optical_NA1.1_lambda525_size79_Z23_upsample1_5000/weights_5000.h5' \
  --model 'twostage_RCAN3D' \
  --background 100 \
  --num_seg_window_x 3 \
  --num_seg_window_y 3 \
  --num_seg_window_z 3 \
  --overlap_x 20 \
  --overlap_y 20 \
  --overlap_z 4 \
  --insert_xy 8 \
  --insert_z 2 \
  --upsample_flag 1 \
  --Fourier_damping_flag 0 \
  --bs 1

echo "Inference completed!"