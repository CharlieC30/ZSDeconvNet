#!/bin/bash
#
# 自訂 3D 訓練腳本 - 按照 ReadMe.md 指南
# 使用我們生成的資料和 PSF
#

# 切換到正確目錄
cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

# 確保使用 TensorFlow 環境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet

echo "開始 ZS-DeconvNet 3D 訓練..."
echo "使用 Python 生成的增強資料"
echo "================================="

# 執行訓練
python Train_ZSDeconvNet_3D.py \
  --psf_path '../../Python_PSF/PSFoutput/other/PSF_XY1.88um_Z15.04um_oddZ_111.tif' \
  --data_dir '../your_augmented_datasets/aisr/' \
  --folder 'aisr122424_roi_0905_1200_100' \
  --test_images_path '../../Raw_Data/FromGary/input/iUExM/roiC_crop128_1128.tif' \
  --save_weights_dir './my_models_3d' \
  --save_weights_suffix '_PSF_XY1.88um_Z15.04um_oddZ_111_upsample0' \
  --model 'twostage_RCAN3D' \
  --upsample_flag 0 \
  --iterations 500 \
  --test_interval 500 \
  --valid_interval 100 \
  --batch_size 2 \
  --gpu_id 1 \
  --gpu_memory_fraction 0.9 \
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
  --Hess_weight 0.1

echo "訓練完成！"