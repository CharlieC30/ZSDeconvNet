# ZS-DeconvNet (TensorFlow Version) Analysis

This document summarizes the analysis of the 2D non-SIM TensorFlow implementation in `Python_MATLAB_Codes/train_inference_python`.

## 1. Execution Workflow

- **Entry Point**: The training is initiated by `train_demo_2D.sh`.
- **Main Script**: The shell script executes `Train_ZSDeconvNet_2D.py`.
- **Dependencies**: The core dependency is `tensorflow-gpu==2.5.0`. Other libraries like `tifffile` and `opencv-python` are used for data handling.

## 2. Model Architecture (`models/twostage_Unet.py`)

- The model is composed of **two full U-Net architectures connected in series**.
- **Stage 1 (Denoising U-Net)**:
    - Takes the raw, noisy image as input.
    - Produces a "denoised" intermediate image (`output1`).
- **Stage 2 (Deconvolution U-Net)**:
    - Takes the "denoised" image (`output1`) from Stage 1 as its input.
    - Produces the final "deconvolved" image (`output2`).
    - An optional `UpSampling2D` layer can be applied for super-resolution, controlled by the `--upsample_flag` argument.
- **Final Model**: The Keras model is packaged with a single input (`inputs`) and two outputs (`output1`, `output2`).

## 3. Training and Loss Function (`Train_ZSDeconvNet_2D.py` & `utils/loss.py`)

- **Dual Loss Calculation**: The model is compiled with two loss functions and corresponding weights.
    - **Loss 1 (Denoising)**: A simple `mean_absolute_error` or `mean_squared_error` between the denoised `output1` and the ground truth `gt_g`.
    - **Loss 2 (Deconvolution)**: A custom loss function, `create_psf_loss`, applied to `output2`.
- **Total Loss**: `total_loss = w * loss_denoise + (1-w) * loss_deconv`, where `w` is `--denoise_loss_weight`.
- **`create_psf_loss` Logic**:
    - **Re-blurring**: It takes the model's deconvolved output (`y_pred`) and convolves it with the system's Point Spread Function (PSF).
    - **Fidelity Term**: It calculates the MAE/MSE between the re-blurred image and the ground truth (`y_true`).
    - **Regularization Terms**: It adds weighted L1, Total Variation (TV), and Hessian regularization terms to the loss. These terms are applied to the model's output `y_pred` to enforce sparsity and smoothness.

## 4. Plan for PyTorch Lightning Migration

- **Model**: We will discard the first U-Net and implement only the **second U-Net** architecture in PyTorch. The input to this model will be the raw image.
- **Loss Function**: We will replicate the `create_psf_loss` logic in PyTorch. The loss will be a combination of the re-blurring fidelity term and the regularization terms.
- **Output**: The PyTorch model will have only a **single output**: the final, deconvolved image.
