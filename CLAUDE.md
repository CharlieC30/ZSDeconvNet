# CLAUDE.md

## Project Overview

ZS-DeconvNet implements self-supervised deep learning for simultaneous denoising and super-resolution in fluorescence microscopy without requiring paired training data.

## Architecture

Four main implementations:
1. **Python/TensorFlow**: Primary implementation in `Python_MATLAB_Codes/train_inference_python/`
2. **MATLAB**: Data preprocessing and PSF generation
3. **Fiji Plugin**: ImageJ/Fiji end-user plugin
4. **PyTorch**: Pure deconvolution implementation in `PyTorch_Deconv/`

## Environment Setup

```bash
conda activate zs-deconvnet_pytorch
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt
conda install cudatoolkit==11.3.1 cudnn==8.2.1
```

## Core Implementation: train_inference_python Directory

### Main Training Scripts

#### `Train_ZSDeconvNet_2D.py` - Comprehensive 2D Training Implementation
- **Purpose**: 2D training for wide-field microscopy with two-stage U-Net
- **Model**: Two independent U-Net architectures (denoising then deconvolution)
- **Training Logic**: Self-supervised with PSF-constrained physics-based loss

##### Model Architecture Parameters:
- `--conv_block_num`: U-Net encoder/decoder depth (default 4, creates 5 resolution levels)
- `--conv_num`: Convolutions per block (default 3, builds deep feature extraction)
- `--upsample_flag`: Enable 2x super-resolution (0/1, affects output size and PSF handling)

##### Training Environment:
- `--gpu_id`: GPU device selection (default 0)
- `--gpu_memory_fraction`: GPU memory limit (default 0.9, prevents OOM)
- `--mixed_precision_training`: Mixed precision for faster training (default 1)
- `--iterations`: Total training steps (default 50000 for 2D wide-field)
- `--test_interval`: Model saving interval (default 1000)
- `--valid_interval`: Validation interval (default 1000)
- `--load_init_model_iter`: Resume training from iteration (default 0)

##### Learning Rate Schedule:
- `--start_lr`: Initial learning rate (default 5e-5, optimized for Adam)
- `--lr_decay_factor`: LR decay multiplier (default 0.5, applied every 10000 iterations)

##### Data Configuration:
- `--otf_or_psf_path`: Path to PSF (.tif) or OTF (.mrc) file
- `--psf_src_mode`: PSF source format (1=PSF .tif, 2=OTF .mrc)
- `--dxypsf`: PSF pixel size in microns (default 0.0313µm)
- `--dx`, `--dy`: Target image pixel size (default 0.0313µm, must be dxypsf/2 if upsample_flag=1)
- `--data_dir`: Root directory containing training folders
- `--folder`: Training data folder name (contains input/ and gt/ subdirectories)
- `--test_images_path`: Path to test images for validation during training

##### Image Processing:
- `--batch_size`: Training batch size (default 4)
- `--input_x`, `--input_y`: Training patch size (default 128x128 pixels)
- `--insert_xy`: Padding margin to avoid edge artifacts (default 16 pixels)
- `--input_x_test`, `--input_y_test`: Test image dimensions (default 512x512)
- `--valid_num`: Number of validation samples (default 3)

##### Loss Function Configuration:
- `--mse_flag`: Loss type (0=MAE, 1=MSE, MAE generally better for images)
- `--denoise_loss_weight`: Balance between denoising and deconvolution stages (default 0.5)
- `--l1_rate`: L1 sparsity regularization weight (default 0, promotes sparse solutions)
- `--TV_rate`: Total Variation regularization (default 0, edge-preserving smoothness)
- `--Hess_rate`: Hessian regularization (default 0.02, second-order smoothness)

#### Other Training Scripts
- **`Train_ZSDeconvNet_2DSIM.py`**: 2D SIM with data augmentation
- **`Train_ZSDeconvNet_3D.py`**: 3D volumetric (wide-field, confocal, LLS), supports twostage_Unet3D and RCAN3D
- **`Train_ZSDeconvNet_3DSIM.py`**: 3D SIM with RCAN3D variants

### Inference Scripts

#### `Infer_2D.py` - Advanced Patch-Based Inference System
- **Purpose**: 2D inference with intelligent memory management and artifact-free reconstruction
- **Algorithm**: Segmentation, Prediction, Fusion pipeline for large images

##### Core Parameters:
- `--input_dir`: Input image path(s) (supports .tif and .mrc formats)
- `--load_weights_path`: Path to trained model weights (.h5 file)
- `--upsample_flag`: Enable 2x super-resolution (0/1, must match training)
- `--insert_xy`: Padding margin (default 16, may be increased automatically)

##### Segmentation Control:
- `--bs`: Batch size per input image (list format: [1] for single image)
- `--num_seg_window_x/y`: Number of patches along each axis (list format: [1])
- `--overlap_x/y`: Overlap between patches in pixels (default [20], minimum 16)

##### Memory Management:
- **Automatic Segmentation**: Calculates optimal patch size based on GPU memory
- **Dynamic Padding**: Adjusts padding to ensure divisibility by 16 (U-Net requirement)
- **Overlap Handling**: Sophisticated fusion algorithm prevents tile artifacts

##### Processing Pipeline:
1. **Image Loading**: Supports TIFF and MRC formats with automatic normalization
2. **Adaptive Segmentation**: Calculates patch grid based on overlap requirements
3. **Padding Calculation**: Ensures each patch meets U-Net size constraints (divisible by 16)
4. **Batch Processing**: Processes patches in configurable batch sizes
5. **Intelligent Fusion**: Combines overlapping predictions with seamless blending

##### Output Structure:
- **Location**: `load_weights_path/../Inference/`
- **Files**: `img{i}_denoised.tif` and `img{i}_deconved.tif`
- **Format**: 16-bit TIFF with 10000x scaling factor for visualization
- **Super-resolution**: Deconvolved output is 2x larger if upsample_flag=1

#### `Infer_3D.py`
3D volumetric inference with patch processing and overlap handling.

### Model Architecture Details

#### `models/twostage_Unet.py` - Two-Stage U-Net Architecture
```python
# Complete Two-Stage Pipeline:
# Input → Stage 1 (Denoising U-Net) → Stage 2 (Deconvolution U-Net) → Outputs
#
# Stage 1: Self-supervised denoising using input reconstruction
# Stage 2: Physics-based deconvolution with PSF constraints
```

##### Architecture Specifications:
- **Input Shape**: `(height, width, 1)` - Single-channel grayscale images
- **Two Independent U-Nets**: Each with identical architecture but different purposes
- **Channel Progression**: 32 → 64 → 128 → 256 → 512 channels (with conv_block_num=4)
- **Skip Connections**: Feature concatenation between encoder and decoder at each level

##### Stage 1 - Denoising U-Net:
```python
# Encoder: 4 downsampling blocks
# - conv_block(32), MaxPool → conv_block(64), MaxPool → ... → conv_block(512)
# - Each conv_block: conv_num convolutions (default 3) with ReLU activation
# Bottleneck: 1024 → 512 channels with 2 convolutions
# Decoder: 4 upsampling blocks with skip connections
# - UpSample + Concatenate + conv_block(channels//2)
# Output: Single-channel denoised image (with padding removal)
```

##### Stage 2 - Deconvolution U-Net:
```python
# Input: Denoised output from Stage 1
# Architecture: Identical encoder-decoder structure
# Key Difference: Optional 2x UpSampling before final convolutions
# Final Layers: 128 → 128 → 1 channel convolutions
# Output: Deconvolved image (potentially 2x super-resolved)
```

##### Key Implementation Details:
- **Padding Strategy**: `insert_x`, `insert_y` parameters remove boundary artifacts
- **Output Cropping**: Stage 1 output cropped to remove padding effects
- **Super-resolution**: Stage 2 conditionally applies 2x upsampling
- **Activation**: ReLU throughout (promotes non-negative intensities)
- **Skip Connections**: Encoder features concatenated to decoder at matching resolutions

##### Mathematical Framework:
```python
# Stage 1: I_denoised = U-Net₁(I_noisy)
# Stage 2: I_deconv = U-Net₂(I_denoised) 
# If upsample_flag=1: I_deconv = UpSample(U-Net₂(I_denoised))
# Output shapes:
#   - output1: [batch, input_y, input_x, 1] (denoised)
#   - output2: [batch, input_y*(1+upsample), input_x*(1+upsample), 1] (deconvolved)
```

#### Other Model Architectures
- **`twostage_Unet3D.py`**: 3D version with Conv3D operations and separate XY/Z padding
- **`twostage_RCAN3D.py`**: RCAN variants with residual channel attention blocks

### Utilities Directory

#### `utils/loss.py` - Physics-Based Loss Functions with Mathematical Detail

##### PSF-Constrained Loss Function (`create_psf_loss`):
```python
# Core Physics-Based Loss: Re-blur Concept
# Predicted Clear Image → PSF Convolution → Re-blurred Image → Compare with Original
# Loss = |I_original - Conv(I_predicted, PSF)| + Regularization Terms

def psf_loss(y_true, y_pred):
    # Stage 1: Re-blur the predicted clear image using known PSF
    y_conv = K.conv2d(y_pred, psf, padding='same')
    
    # Stage 2: Handle super-resolution scaling
    if upsample_flag:
        y_conv = tf.image.resize(y_conv, [height//2, width//2])
    
    # Stage 3: Remove padding artifacts
    y_conv = y_conv[:, insert_xy:-insert_xy, insert_xy:-insert_xy, :]
    
    # Stage 4: Compute reconstruction loss (MAE or MSE)
    reconstruction_loss = K.mean(K.abs(y_true - y_conv))  # MAE by default
    
    return reconstruction_loss + regularization_terms
```

##### Regularization Components:

**Total Variation (TV) Loss**:
```python
# Promotes smooth, edge-preserving solutions
# TV_loss = Σ|∇x I| + Σ|∇y I| (L2 norm of gradients)
y_diff = image[:, :-1, :] - image[:, 1:, :]  # Vertical gradients
x_diff = image[:, :, :-1] - image[:, :, 1:]  # Horizontal gradients
TV_loss = tf.nn.l2_loss(x_diff) + tf.nn.l2_loss(y_diff)
```

**Hessian Loss**:
```python
# Second-order smoothness penalty (discourages fine-scale artifacts)
# Hess_loss = Σ|∇²I| = |∂²I/∂x²| + |∂²I/∂y²| + 2|∂²I/∂x∂y|
xx = x_grad[:, :, :-1] - x_grad[:, :, 1:]    # ∂²I/∂x²
yy = y_grad[:, :-1, :] - y_grad[:, 1:, :]    # ∂²I/∂y²
xy = y_grad[:, :, :-1] - y_grad[:, :, 1:]    # ∂²I/∂x∂y
yx = x_grad[:, :-1, :] - x_grad[:, 1:, :]    # ∂²I/∂y∂x
Hess_loss = l2_loss(xx) + l2_loss(yy) + l2_loss(xy) + l2_loss(yx)
```

**L1 Sparsity Loss**:
```python
# Promotes sparse (zero-heavy) solutions, reduces background noise
L1_loss = K.mean(K.abs(y_pred))
```

##### Complete Loss Combination:
```python
total_loss = reconstruction_loss + 
             TV_weight * TV_loss + 
             Hess_weight * Hess_loss + 
             l1_rate * L1_loss
```

##### PSF Processing Pipeline:
**PSF Estimation and Cropping**:
```python
# 1. Estimate PSF standard deviation for optimal cropping
sigma_y, sigma_x = psf_estimator_2d(psf)
ksize = int(sigma_y * 4)  # Crop to 4-sigma radius (99.99% energy)

# 2. Crop PSF to reduce computational cost
if ksize <= half_width:
    psf_cropped = psf[center-ksize:center+ksize+1, center-ksize:center+ksize+1]

# 3. Normalize PSF for proper convolution
psf_normalized = psf_cropped / np.sum(psf_cropped)

# 4. Reshape for TensorFlow convolution: (height, width, in_channels, out_channels)
psf_tensor = psf_normalized.reshape(height, width, 1, 1).astype(np.float32)
```

##### Two-Stage Loss Application:
```python
# Training uses both stages with weighted combination:
# Stage 1 (Denoising): MAE/MSE loss for input reconstruction
# Stage 2 (Deconvolution): PSF loss with regularization

loss_functions = [
    'mean_absolute_error',  # Stage 1: Simple reconstruction
    psf_loss               # Stage 2: Physics-constrained deconvolution
]

loss_weights = [denoise_loss_weight, 1 - denoise_loss_weight]
# Typical: [0.5, 0.5] balances denoising and deconvolution
```

#### `utils/data_loader.py` - Intelligent Data Loading with Multi-Modal Support
```python
def DataLoader(images_path, data_path, gt_path, batch_size, norm_flag):
    # Random sampling without replacement for each batch
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    
    # Automatic path mapping: input/image.tif → gt/image.tif
    # Handles missing ground truth gracefully with re-sampling
    
    # Multi-format support: TIFF multi-frame via imageio.mimread()
    # Automatic type conversion to float32 for training
    
    # Normalization strategies:
    # norm_flag=0: /65535 (16-bit TIFF standard)
    # norm_flag=1: prctile_norm() (0-100 percentile, robust to outliers)
    # norm_flag=2: /max() (simple max normalization)
```

##### Normalization Details:
- **norm_flag=0**: Direct division by 65535 (assumes 16-bit TIFF input)
- **norm_flag=1**: Percentile-based robust normalization (recommended)
- **norm_flag=2**: Simple max normalization (may be sensitive to bright outliers)

#### `utils/utils.py` - Core Utility Functions
```python
def prctile_norm(x, min_prc=0, max_prc=100):
    # Robust percentile-based normalization
    # Maps [min_percentile, max_percentile] → [0, 1]
    # Clips values outside range, handles divide-by-zero
    # Default: 0-100 percentile (full range)
    # Common: 3-100 percentile (removes dark noise)

def read_mrc(filename, filetype='image'):
    # Comprehensive MRC (Medical Research Council) format reader
    # Supports complex header parsing for OTF files
    # Handles byte order (endianness) detection
    # Returns (header, data) tuple for full metadata access
    # Essential for reading optical transfer functions from microscopy software
```

##### MRC Format Support:
- **Header Parsing**: Complete MRC header structure with 256-byte standard
- **Data Types**: Supports mode 0-4 (various integer and float formats)
- **Endianness**: Automatic detection via header stamp bytes
- **Metadata**: Pixel size, dimensions, intensity scaling information

#### Other Utilities
- **`utils/augment_sim_img.py`**: SIM-specific data augmentation with 8-fold rotation/flipping

### Demo Scripts
- **Training**: `train_demo_2D.sh`, `train_demo_2DSIM.sh`, `train_demo_3D.sh`, `train_demo_3DSIM.sh`
- **Inference**: `infer_demo_2D.sh`, `infer_demo_3D.sh`

### Comprehensive Training Workflow - Step-by-Step Implementation

#### Phase 1: Data Preparation and Organization
```
data_dir/folder/
├── input/          # Training input images (blurred/noisy microscopy data)
│   ├── cell01.tif  # Individual TIFF files or multi-frame stacks
│   ├── cell02.tif
│   └── ...
└── gt/             # Ground truth images (clean reference, if available)
    ├── cell01.tif  # Often identical to input/ for self-supervised training
    ├── cell02.tif
    └── ...
```

#### Phase 2: PSF/OTF Processing Pipeline
```python
# PSF Mode Selection (psf_src_mode):
# 1 = PSF in .tif format (theoretical or measured point spread function)
# 2 = OTF in .mrc format (optical transfer function from microscopy software)

if psf_src_mode == 1:  # PSF .tif processing
    # 1. Load PSF image and determine pixel scaling
    # 2. Interpolate PSF to match target resolution (dxypsf → dx)
    # 3. Handle odd/even dimensions with careful interpolation
    # 4. Convert PSF → OTF via FFT
    # 5. Resize OTF to match training image dimensions
    
elif psf_src_mode == 2:  # OTF .mrc processing  
    # 1. Read OTF from MRC file using custom reader
    # 2. Calculate PSF via inverse FFT
    # 3. Adjust dimensions for super-resolution (if upsample_flag=1)
    # 4. Apply appropriate k-space sampling (dkx, dky)

# PSF Optimization for Training:
# 1. Estimate PSF standard deviation (sigma estimation)
# 2. Crop PSF to 4-sigma radius (99.99% energy retention)
# 3. Normalize PSF (sum = 1) for proper convolution
# 4. Save processed PSF and OTF for visualization
```

#### Phase 3: Model Architecture Instantiation
```python
# Create two model instances:
# g: Training model with training patch dimensions
# p: Testing model with full test image dimensions

g = Unet((input_y + 2*insert_xy, input_x + 2*insert_xy, 1),
         upsample_flag=upsample_flag, 
         insert_x=insert_xy, insert_y=insert_xy)

p = Unet((input_y_test + 2*insert_xy_test, input_x_test + 2*insert_xy_test, 1),
         upsample_flag=upsample_flag, 
         insert_x=insert_xy_test, insert_y=insert_xy_test)
```

#### Phase 4: Loss Function Compilation and Optimization
```python
# Multi-loss training setup:
psf_loss_fn = create_psf_loss(psf_tensor, TV_rate, Hess_rate, 
                              laplace_weight=0, l1_rate, mse_flag, 
                              upsample_flag, insert_xy, deconv_flag=1)

loss_functions = ['mean_absolute_error', psf_loss_fn]
loss_weights = [denoise_loss_weight, 1 - denoise_loss_weight]

# Adam optimizer with learning rate scheduling:
optimizer = Adam(lr=start_lr, beta_1=0.9, beta_2=0.999)
g.compile(loss=loss_functions, loss_weights=loss_weights, optimizer=optimizer)
```

#### Phase 5: Training Loop with Validation and Monitoring
```python
for iteration in range(iterations):
    # 1. Load random batch from training data
    [input_batch, gt_batch] = DataLoader(...)
    
    # 2. Apply padding to avoid boundary artifacts
    input_padded = add_padding(input_batch, insert_xy)
    
    # 3. Train on batch (both outputs use same ground truth)
    loss = g.train_on_batch(x=input_padded, y=[gt_batch, gt_batch])
    
    # 4. Learning rate decay schedule (every 10000 iterations)
    if (iteration + 1) % 10000 == 0:
        lr = lr * lr_decay_factor
        K.set_value(g.optimizer.learning_rate, lr)
    
    # 5. Validation and testing (every 1000 iterations)
    if (iteration + 1) % valid_interval == 0:
        validate_on_training_samples()
        save_model_weights()
        test_on_held_out_images()
    
    # 6. TensorBoard logging
    log_losses_and_learning_rate()
```

#### Phase 6: PSF Processing Mathematical Details
```python
# PSF Interpolation (when dxypsf ≠ dx):
sr_ratio = dxypsf / dx  # Super-resolution scaling factor

# For odd PSF dimensions (preferred):
new_width = round(psf_width * sr_ratio)
if new_width % 2 == 0:  # Ensure odd dimensions
    new_width += 1 if new_width < psf_width * sr_ratio else -1

# For even PSF dimensions (complex interpolation):
# Split into left/right halves, interpolate separately, recombine
# Ensures PSF center remains properly defined

# PSF Energy Conservation:
psf_normalized = psf_interpolated / np.sum(psf_interpolated)
```

#### Phase 7: Monitoring and Output Management
```python
# Directory structure created automatically:
save_weights_path/
├── config.txt              # Complete parameter configuration
├── psf.tif                 # Processed PSF for verification
├── otf.tif                 # Computed OTF for verification
├── weights_1000.h5         # Model checkpoints
├── weights_2000.h5
├── TrainSampled/           # Validation results during training
│   ├── input_sample_0.tif
│   ├── 0denoised_iter01000.tif
│   └── 0deconved_iter01000.tif
├── TestSampled/            # Test results during training
└── graph/                  # TensorBoard logs
```

### Advanced Inference Workflow - Production-Ready Processing

#### Stage 1: Intelligent Image Preparation
```python
# Multi-format input support:
if 'tif' in input_path:
    image = tiff.imread(input_path).astype('float')
elif 'mrc' in input_path:
    header, image = read_mrc(input_path)
    image = image.transpose((1, 0))  # Correct axis order

# Robust preprocessing:
image[image < 0] = 0  # Remove negative values (common in MRC)
image = prctile_norm(image)  # Percentile normalization (0-1 range)
```

#### Stage 2: Adaptive Segmentation Algorithm
```python
# Calculate optimal patch dimensions:
seg_window_x = ceil((inp_x + (num_seg_x-1) * overlap_x) / num_seg_x)
seg_window_y = ceil((inp_y + (num_seg_y-1) * overlap_y) / num_seg_y)

# Ensure U-Net compatibility (divisible by 16):
conv_block_num = 4  # U-Net depth
required_factor = 2 ** conv_block_num  # 16 for 4-level U-Net

# Calculate dynamic padding:
n = ceil(seg_window_x / required_factor)
while required_factor * n - seg_window_x < 2 * insert_xy:
    n += 1
insert_x = int((required_factor * n - seg_window_x) / 2)

# Generate patch coordinates:
rr_list = range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x)
if rr_list[-1] != inp_x - seg_window_x:
    rr_list.append(inp_x - seg_window_x)  # Ensure complete coverage
```

#### Stage 3: Batch Processing with Memory Management
```python
# Create model with exact patch dimensions:
model = Unet([seg_window_x + 2*insert_x, seg_window_y + 2*insert_y, 1],
             upsample_flag=upsample_flag, insert_x=insert_x, insert_y=insert_y)
model.load_weights(weights_path)

# Process in configurable batches:
for batch_start in batch_indices:
    batch_end = min(batch_start + batch_size, total_patches)
    batch_patches = segmented_patches[batch_start:batch_end]
    
    # Predict both outputs:
    predictions = model.predict(batch_patches)
    denoised_outputs = predictions[0]  # Stage 1 output
    deconvolved_outputs = predictions[1]  # Stage 2 output
    
    # Handle super-resolution cropping:
    if upsample_flag:
        deconv_cropped = deconvolved_outputs[
            :, insert_x*2:(seg_window_x+insert_x)*2, 
               insert_y*2:(seg_window_y+insert_y)*2]
    else:
        deconv_cropped = deconvolved_outputs[
            :, insert_x:seg_window_x+insert_x,
               insert_y:seg_window_y+insert_y]
```

#### Stage 4: Seamless Patch Fusion Algorithm
```python
# Initialize output arrays:
output_deconv = zeros((inp_x*(1+upsample_flag), inp_y*(1+upsample_flag)))
output_denoise = zeros((inp_x, inp_y))

# Sophisticated overlap handling:
for r_idx, row_start in enumerate(rr_list):
    for c_idx, col_start in enumerate(cc_list):
        
        # Calculate fusion boundaries (avoid edge artifacts):
        if row_start == 0:
            row_min = 0; row_min_patch = 0
        else:
            row_min = row_start + ceil(overlap_x/2)
            row_min_patch = ceil(overlap_x/2)
        
        if row_start + seg_window_x == inp_x:
            row_max = inp_x; row_max_patch = seg_window_x
        else:
            row_max = row_start + seg_window_x - floor(overlap_x/2)
            row_max_patch = seg_window_x - floor(overlap_x/2)
        
        # Apply fusion with proper scaling:
        patch_idx = r_idx * len(cc_list) + c_idx
        
        # Deconvolved (potentially super-resolved):
        deconv_patch = dec_list[patch_idx, 
            row_min_patch*(1+upsample_flag):row_max_patch*(1+upsample_flag),
            col_min_patch*(1+upsample_flag):col_max_patch*(1+upsample_flag)]
        
        output_deconv[
            row_min*(1+upsample_flag):row_max*(1+upsample_flag),
            col_min*(1+upsample_flag):col_max*(1+upsample_flag)] = deconv_patch
```

#### Stage 5: Output Processing and Saving
```python
# Convert to 16-bit with scaling for visualization:
output_deconv_16bit = uint16(10000 * prctile_norm(output_deconv, 3, 100))
output_denoise_16bit = uint16(10000 * prctile_norm(output_denoise, 3, 100))

# Save with descriptive filenames:
save_path = dirname(weights_path) + '/Inference/'
tiff.imwrite(save_path + f'img{i}_denoised.tif', output_denoise_16bit)
tiff.imwrite(save_path + f'img{i}_deconved.tif', output_deconv_16bit)
```

## Practical Implementation Examples for 2D Wide-Field Microscopy

### Complete Training Example
```bash
# Optimized parameters for 2D wide-field deconvolution
python Train_ZSDeconvNet_2D.py \
    --otf_or_psf_path '/path/to/microscope_PSF.mrc' \
    --data_dir '/path/to/training/data/' \
    --folder 'experiment_widefield_cells' \
    --test_images_path '/path/to/test/validation_image.tif' \
    --psf_src_mode 2 \
    --iterations 50000 \
    --batch_size 4 \
    --start_lr 5e-5 \
    --lr_decay_factor 0.5 \
    --input_x 128 --input_y 128 \
    --insert_xy 16 \
    --upsample_flag 1 \
    --denoise_loss_weight 0.5 \
    --TV_rate 0 \
    --Hess_rate 0.02 \
    --l1_rate 0 \
    --mse_flag 0 \
    --save_weights_dir '../saved_models/' \
    --save_weights_suffix '_2D_widefield'
```

### Complete Inference Example
```bash
# Single image inference with automatic segmentation
python Infer_2D.py \
    --input_dir '/path/to/test/large_image.tif' \
    --load_weights_path '../saved_models/experiment/weights_50000.h5' \
    --num_seg_window_x 4 \
    --num_seg_window_y 4 \
    --overlap_x 32 \
    --overlap_y 32 \
    --bs 1 \
    --insert_xy 16 \
    --upsample_flag 1

# Multiple image batch processing
python Infer_2D.py \
    --input_dir '/path/to/test/*.tif' \
    --load_weights_path '../saved_models/experiment/weights_50000.h5' \
    --num_seg_window_x 1 1 1 \
    --num_seg_window_y 1 1 1 \
    --overlap_x 20 20 20 \
    --overlap_y 20 20 20 \
    --bs 2 2 2
```

### Parameter Tuning Guidelines

#### Training Parameters:
- **iterations**: 50000 for wide-field (2D), 10000 for 3D volumes
- **batch_size**: 4-8 for 2D (128×128 patches), 1-2 for 3D
- **start_lr**: 5e-5 (2D), 1e-4 (3D) - Adam optimizer optimized
- **input_x/y**: 128 (training patches), 512+ (test images)
- **insert_xy**: 16 (2D), 8 (3D) - balance between artifact removal and computation

#### Loss Function Weights:
- **denoise_loss_weight**: 0.5 (equal denoising and deconvolution)
- **TV_rate**: 0 (2D wide-field), 0.1 (SIM, noisy data)
- **Hess_rate**: 0.02 (standard), 0.1 (strong artifact suppression)
- **l1_rate**: 0 (typically unused for microscopy)

#### Inference Parameters:
- **num_seg_window_x/y**: Start with 1, increase for large images or GPU memory limits
- **overlap_x/y**: Minimum 20 pixels, increase for seamless fusion
- **bs**: 1-4 depending on patch size and GPU memory

### Other Parameter Sets
- **3D**: `--iterations 10000`, `--batch_size 1`, `--start_lr 1e-4`, `--insert_xy 8 --insert_z 2`
- **SIM**: `--augment_flag True`, `--TV_rate 0.1`

### Monitoring and Output
- **TensorBoard**: `tensorboard --logdir <save_weights_dir>/<save_weights_name>/graph`
- **Output Structure**: `saved_models/[experiment]/` contains `saved_model/`, `Inference/`, `graph/`, `test_data/`

### Performance Tips
- Large images: Use `num_seg_window_x/y` for segmentation
- Overlap: Minimum 16 pixels to avoid artifacts  
- Memory: Adjust `--gpu_memory_fraction` and batch size

# PyTorch Lightning Implementation

Alternative pure deconvolution implementation in `PyTorch_Deconv/`. 

## Key Differences
- Pure deconvolution only (removes denoising stage)
- Self-supervised training: input = target = blurred image
- PSF-constrained physics-based loss
- 2x super-resolution capability

## Current Status
Working directory: `/home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_Deconv/`
Environment: `zs-deconvnet_pytorch`