#!/bin/bash
#
# Custom Inference Script - PyTorch Version
# Based on your existing TensorFlow infer_custom_3d.sh
# Using your trained model
#

# Switch to correct directory
cd /home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_3d

# Ensure using PyTorch environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet_pytorch

echo "Starting inference with trained ZS-DeconvNet model (PyTorch Version)..."
echo "Using model: twostage_RCAN3D"
echo "Weights: model_weights_500.pth (final training result)"
echo "======================================================================="

# Execute inference with your custom parameters
python Infer_3D.py \
  --input_dir 'data/ori_input/aisr/aisr122424_roi.tif' \
  --load_weights_path './my_models_3d/model_weights_500.pth' \
  --output_dir './inference_results' \
  --model 'twostage_RCAN3D' \
  --background 100 \
  --use_tiling \
  --tile_size_x 256 \
  --tile_size_y 256 \
  --tile_size_z 64 \
  --overlap_x 20 \
  --overlap_y 20 \
  --overlap_z 4 \
  --insert_xy 8 \
  --insert_z 2 \
  --upsample_flag 0 \
  --device auto

echo "Inference completed!"
echo "Results saved in: ./inference_results/"
echo "Check for:"
echo "  - *_denoised.tif: Denoised results"
echo "  - *_deconvolved.tif: Final deconvolved results"