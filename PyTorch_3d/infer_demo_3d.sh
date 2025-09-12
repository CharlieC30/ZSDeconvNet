#!/bin/bash
#
# ZS-DeconvNet 3D Inference Demo Script - PyTorch Version
# Equivalent to original TensorFlow infer_demo_3D.sh
#

# ------------------------------- Inference Arguments -------------------------------

# Tiling parameters (for large images)
# --use_tiling: Enable tiling for large images
# --tile_size_x: tile size along x axis
# --tile_size_y: tile size along y axis  
# --tile_size_z: tile size along z axis
# --overlap_x: overlapping in x direction. NOTICE: if tiling enabled, overlap should be big enough to avoid artifacts
# --overlap_y: overlapping in y direction
# --overlap_z: overlapping in z direction

# Input/Output
# --input_dir: the root path of test data (can be single file or directory)
# --output_dir: output directory for results (optional, defaults to model directory)
# --load_weights_path: path to trained model weights (.pth file)
# --background: the background value you want to subtract from input images

# Model settings
# --model: "twostage_RCAN3D" or "twostage_Unet3D"
# --insert_xy: padded blank margin in pixels in each side 
# --insert_z: padded blank margin in each side of axial direction (how many slices)
# --upsample_flag: 0 or 1, whether the network upsamples the image

# PyTorch specific
# --device: 'auto', 'cpu', or 'cuda' for device selection

cd /home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_3d

# Activate PyTorch environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet_pytorch

# ------------------------------- Examples -------------------------------

echo "Starting ZS-DeconvNet 3D Inference (PyTorch Version)..."
echo "======================================================"

# Example 1: Simple inference without tiling
echo "Example 1: Simple inference without tiling"
python Infer_3D.py \
    --model 'twostage_RCAN3D' \
    --load_weights_path './models/your_model_weights.pth' \
    --input_dir 'path/to/your/input_image.tif' \
    --output_dir './inference_results' \
    --background 100 \
    --insert_xy 8 \
    --insert_z 2 \
    --upsample_flag 0 \
    --device auto

echo ""
echo "Example 2: Inference with tiling for large images"
python Infer_3D.py \
    --model 'twostage_RCAN3D' \
    --load_weights_path './models/your_model_weights.pth' \
    --input_dir 'path/to/your/large_image.tif' \
    --output_dir './inference_results_tiling' \
    --background 100 \
    --use_tiling \
    --tile_size_x 256 \
    --tile_size_y 256 \
    --tile_size_z 64 \
    --overlap_x 32 \
    --overlap_y 32 \
    --overlap_z 8 \
    --insert_xy 8 \
    --insert_z 2 \
    --upsample_flag 0 \
    --device auto

echo ""
echo "Example 3: Batch processing multiple images"
python Infer_3D.py \
    --model 'twostage_RCAN3D' \
    --load_weights_path './models/your_model_weights.pth' \
    --input_dir 'path/to/your/image_directory/' \
    --output_dir './inference_results_batch' \
    --background 0 \
    --insert_xy 8 \
    --insert_z 2 \
    --upsample_flag 0 \
    --device auto

echo ""
echo "Example 4: Super-resolution inference"
python Infer_3D.py \
    --model 'twostage_RCAN3D' \
    --load_weights_path './models/your_superres_model_weights.pth' \
    --input_dir 'path/to/your/input.tif' \
    --output_dir './inference_results_superres' \
    --background 100 \
    --use_tiling \
    --tile_size_x 128 \
    --tile_size_y 128 \
    --tile_size_z 32 \
    --overlap_x 16 \
    --overlap_y 16 \
    --overlap_z 4 \
    --insert_xy 8 \
    --insert_z 2 \
    --upsample_flag 1 \
    --device auto

echo "Inference completed! Check output directories for results."
echo "Results include:"
echo "  - *_denoised.tif: Denoised images"
echo "  - *_deconvolved.tif: Deconvolved images (main output)"