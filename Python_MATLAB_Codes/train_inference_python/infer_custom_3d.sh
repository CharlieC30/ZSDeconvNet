#!/bin/bash
#
# 自訂推論腳本 - 使用我們訓練好的模型
# 按照 ReadMe.md 第5部分指引
#

# 切換到正確目錄
cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

# 確保使用 TensorFlow 環境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet

echo "開始使用訓練好的 ZS-DeconvNet 模型進行推論..."
echo "使用模型: twostage_RCAN3D"
echo "權重: weights_500.h5 (最終訓練結果)"
echo "========================================"

# 重置 GPU 狀態
export TF_FORCE_GPU_ALLOW_GROWTH=true
# export CUDA_VISIBLE_DEVICES=1

# 執行推論
python Infer_3D.py \
  --input_dir 'data/ori_input/aisr/aisr122424_roi.tif' \
  --load_weights_path './my_models_3d/aisr122424_roi_0905_1200_100_twostage_RCAN3D_PSF_XY1.88um_Z15.04um_oddZ_111_upsample0/weights_500.h5' \
  --model 'twostage_RCAN3D' \
  --background 100 \
  --num_seg_window_x 4 \
  --num_seg_window_y 4 \
  --num_seg_window_z 4 \
  --overlap_x 20 \
  --overlap_y 20 \
  --overlap_z 4 \
  --insert_xy 8 \
  --insert_z 2 \
  --upsample_flag 0 \
  --Fourier_damping_flag 0

echo "推論完成！"
