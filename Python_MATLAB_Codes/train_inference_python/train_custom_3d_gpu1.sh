cd /home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zs-deconvnet

python Train_ZSDeconvNet_3D.py \
  --psf_path '../../Python_PSF/PSFoutput/other/PSF_XY1.88um_Z15.04um_oddZ_118.tif' \
  --data_dir '../your_augmented_datasets/iUExM/' \
  --folder 'iUExM_roi_0916_1550_100' \
  --test_images_path '../../Raw_Data/FromGary/input/iUExM/roiC_crop128_1128.tif' \
  --save_weights_dir './my_models_3d' \
  --save_weights_suffix '_PSF_XY1.88um_Z15.04um_oddZ_118_upsample0' \
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
  --dx 1 \
  --dz 2 \
  --dxpsf 1 \
  --dzpsf 8 \
  --background 100 \
  --mse_flag 0 \
  --TV_weight 0.0 \
  --Hess_weight 0.1

echo "training done!"