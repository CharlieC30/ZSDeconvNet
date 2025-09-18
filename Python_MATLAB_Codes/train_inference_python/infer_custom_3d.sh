cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet

export TF_FORCE_GPU_ALLOW_GROWTH=true
# export CUDA_VISIBLE_DEVICES=1

python Infer_3D.py \
  --input_dir 'data/ori_input/iUExM/iUExM_roi.tif' \
  --load_weights_path './my_models_3d/iUExM_roi_0916_1550_100_twostage_RCAN3D_PSF_XY1.88um_Z15.04um_oddZ_118_upsample0/weights_500.h5' \
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

echo "inference done!"
