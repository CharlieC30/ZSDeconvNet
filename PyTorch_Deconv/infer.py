import os
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import tifffile
from tqdm import tqdm
import glob

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lightning_module import DeconvolutionLightningModule
from src.utils.psf_utils import prctile_norm


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from various possible locations
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        model_config = hyper_params.get('model_config')
        if model_config is None:
            # Try to extract from other hyper_parameters
            model_config = {
                'input_channels': hyper_params.get('input_channels', 1),
                'output_channels': hyper_params.get('output_channels', 1),
                'conv_block_num': hyper_params.get('conv_block_num', 4),
                'conv_num': hyper_params.get('conv_num', 3),
                'upsample_flag': hyper_params.get('upsample_flag', True),
                'insert_xy': hyper_params.get('insert_xy', 16)
            }
    else:
        model_config = checkpoint.get('model_config')
    
    # If still no model config found, use default
    if model_config is None:
        model_config = {
            'input_channels': 1,
            'output_channels': 1,
            'conv_block_num': 4,
            'conv_num': 3,
            'upsample_flag': True,
            'insert_xy': 16
        }
        print("Warning: Using default model configuration")
    
    # Auto-detect architecture from checkpoint
    architecture = None
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        if 'model_config' in hyper_params and isinstance(hyper_params['model_config'], dict):
            architecture = hyper_params['model_config'].get('architecture', 'deconv_only')
    
    # Check for two-stage model by looking at state dict keys
    if architecture is None and 'state_dict' in checkpoint:
        state_dict_keys = checkpoint['state_dict'].keys()
        if any('stage1_' in key or 'stage2_' in key for key in state_dict_keys):
            architecture = 'two_stage'
        else:
            architecture = 'deconv_only'
    
    print(f"Detected model architecture: {architecture}")
    
    # Load the appropriate Lightning module
    try:
        if architecture == 'two_stage':
            from src.models.two_stage_lightning_module import TwoStageDeconvolutionLightningModule
            model = TwoStageDeconvolutionLightningModule.load_from_checkpoint(
                checkpoint_path, 
                map_location=device,
                strict=False
            )
            print("Successfully loaded Two-Stage Lightning module")
        else:
            from src.models.lightning_module import DeconvolutionLightningModule
            model = DeconvolutionLightningModule.load_from_checkpoint(
                checkpoint_path, 
                map_location=device,
                strict=False
            )
            print("Successfully loaded Deconv-Only Lightning module")
        
        model.eval()
        model.to(device)
        
        # DEBUG: Check if model weights are loaded correctly
        total_params = sum(p.numel() for p in model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
        print(f"DEBUG - Model has {total_params:,} total parameters, {non_zero_params:,} non-zero")
        
        # Extract model config from the loaded model
        if hasattr(model, 'model_config'):
            model_config = model.model_config
        else:
            model_config = model.hparams.get('model_config', model_config)
        
        return model, model_config
        
    except Exception as e:
        print(f"Failed to load Lightning module: {e}")
        print("Falling back to manual model loading...")
        
        # Fallback to manual loading based on detected architecture
        if architecture == 'two_stage':
            from src.models.two_stage_unet import TwoStageUNet
            # Remove architecture from config if present
            model_config_filtered = {k: v for k, v in model_config.items() if k != 'architecture'}
            model = TwoStageUNet(**model_config_filtered)
        else:
            from src.models.deconv_unet import DeconvUNet
            model = DeconvUNet(**model_config)
        
        # Load state dict - only load model weights
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract only model weights (remove 'model.' prefix)
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                model_state_dict[new_key] = value
        
        # Load weights
        try:
            model.load_state_dict(model_state_dict)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            # Try alternative loading method
            print("Trying alternative loading method...")
            model.load_state_dict(model_state_dict, strict=False)
        
        model.eval()
        model.to(device)
        
        return model, model_config


def process_single_image(model, model_config, image_path, output_dir, device='cpu', 
                        tile_size=None, overlap=64, batch_size=1, args=None):
    """
    Process a single 3D TIFF image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_dir: Output directory
        device: Device to run inference on
        tile_size: Size for tiling large images (None = process whole image)
        overlap: Overlap between tiles
        batch_size: Batch size for processing slices
    """
    print(f"Processing: {image_path}")
    
    # Load image
    img = tifffile.imread(image_path).astype(np.float32)
    
    # Handle 2D vs 3D
    if len(img.shape) == 2:
        # 2D image - add slice dimension
        img = img[np.newaxis, ...]
        is_2d = True
    else:
        is_2d = False
    
    num_slices, height, width = img.shape
    print(f"Image shape: {img.shape}")
    
    # Prepare for processing
    insert_xy = model_config.get('insert_xy', 16)
    
    # Process slices
    processed_slices = []
    
    for slice_idx in tqdm(range(num_slices), desc="Processing slices"):
        slice_img = img[slice_idx]
        
        if hasattr(args, 'use_original_tiling') and args.use_original_tiling:
            # Use original tiling method
            processed_slice = process_slice_original_tiling(
                model, slice_img, args.num_seg_window_x, args.num_seg_window_y,
                args.overlap_x, args.overlap_y, insert_xy, device, model_config
            )
        elif tile_size is None or (height <= tile_size and width <= tile_size):
            # Process whole slice
            processed_slice = process_slice_whole(model, slice_img, insert_xy, device)
        else:
            # Process with tiling
            processed_slice = process_slice_tiled(
                model, slice_img, tile_size, overlap, insert_xy, device, model_config
            )
        
        processed_slices.append(processed_slice)
    
    # Stack processed slices
    if is_2d:
        result = processed_slices[0]
    else:
        result = np.stack(processed_slices, axis=0)
    
    # Save result
    output_path = output_dir / f"{Path(image_path).stem}_deconvolved.tif"
    
    print(f"Result shape: {result.shape}, range: [{result.min():.4f}, {result.max():.4f}]")
    
    # Convert to uint16 for saving - use consistent normalization
    if result.max() > result.min():
        # Percentile normalization
        result_norm = prctile_norm(result)
        result_uint16 = (result_norm * 65535).astype(np.uint16)
    else:
        print("Warning: Result has no dynamic range - might be all zeros!")
        result_uint16 = (result * 65535).astype(np.uint16)
    
    tifffile.imwrite(output_path, result_uint16)
    
    print(f"Saved to: {output_path}")
    return output_path


def process_slice_whole(model, slice_img, insert_xy, device):
    """Process a single slice without tiling."""
    # Apply exact same normalization as training (matching datamodule._normalize_image)
    min_val = np.percentile(slice_img, 0)
    max_val = np.percentile(slice_img, 100)
    slice_norm = (slice_img - min_val) / (max_val - min_val + 1e-7)
    slice_norm = np.clip(slice_norm, 0, 1)
    
    # Add padding
    pad_width = ((insert_xy, insert_xy), (insert_xy, insert_xy))
    padded_img = np.pad(slice_norm, pad_width, mode='constant', constant_values=0)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        # Try using the underlying U-Net model directly
        if hasattr(model, 'model'):
            # Use the U-Net model inside Lightning module
            raw_output = model.model(input_tensor)
            
            # For two-stage model, take the second output (deconvolved)
            if isinstance(raw_output, (list, tuple)):
                output = raw_output[-1]  # Take the deconvolved output
                print(f"Two-stage model: using deconvolved output (stage 2)")
            else:
                output = raw_output
        else:
            # Use Lightning module forward
            output = model(input_tensor)
    
    # Convert back to numpy
    result = output.squeeze().cpu().numpy()
    
    # Check and handle output range issues
    if result.max() - result.min() < 0.01:
        print(f"Warning: Output has very narrow range [{result.min():.6f}, {result.max():.6f}]")
        # Apply contrast stretching
        result = (result - result.min()) / (result.max() - result.min() + 1e-7)
        print(f"Applied contrast stretching, new range: [{result.min():.6f}, {result.max():.6f}]")
    
    # Clear GPU memory
    del input_tensor, output
    torch.cuda.empty_cache()
    
    # For 2x super-resolution: expected output should be 2x original size
    original_h, original_w = slice_img.shape
    expected_h, expected_w = original_h * 2, original_w * 2
    
    result_h, result_w = result.shape
    
    # Handle cropping for 2x super-resolution
    if result_h > expected_h or result_w > expected_w:
        # Calculate center crop coordinates
        crop_h_start = (result_h - expected_h) // 2
        crop_w_start = (result_w - expected_w) // 2
        crop_h_end = crop_h_start + expected_h
        crop_w_end = crop_w_start + expected_w
        
        # Ensure valid crop coordinates
        crop_h_start = max(0, crop_h_start)
        crop_w_start = max(0, crop_w_start)
        crop_h_end = min(result_h, crop_h_end)
        crop_w_end = min(result_w, crop_w_end)
        
        result = result[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    
    # If result is smaller than expected, resize up
    elif result_h < expected_h or result_w < expected_w:
        print(f"Warning: Output size ({result_h}, {result_w}) smaller than expected ({expected_h}, {expected_w})")
        import cv2
        result = cv2.resize(result, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    
    return result


def process_slice_original_tiling(model, slice_img, num_seg_window_x, num_seg_window_y, 
                                 overlap_x, overlap_y, insert_xy, device, model_config=None):
    """Process a slice using original tiling method - exact replica of original implementation."""
    import math
    
    inp_x, inp_y = slice_img.shape
    upsample_flag = 1  # Always 2x upsampling
    
    # Calculate segment window sizes (exact replica of original logic)
    seg_window_x = math.ceil((inp_x + (num_seg_window_x - 1) * overlap_x) / num_seg_window_x)
    seg_window_y = math.ceil((inp_y + (num_seg_window_y - 1) * overlap_y) / num_seg_window_y)
    
    # Dynamic insert calculation (modified to ensure total size is multiple of 16)
    conv_block_num = 4
    
    # Calculate insert_x to make total width multiple of 16
    n = math.ceil(seg_window_x / 2**conv_block_num)
    while 16 * n - seg_window_x < 2 * insert_xy:
        n = n + 1
    insert_x = int((16 * n - seg_window_x) / 2)
    
    # Ensure final width (seg_window_x + 2*insert_x) is multiple of 16
    total_width = seg_window_x + 2 * insert_x
    
    if total_width % 16 != 0:
        # Calculate target width (next multiple of 16)
        target_width = ((total_width // 16) + 1) * 16
        padding_needed = target_width - seg_window_x
        insert_x = padding_needed // 2
    
    # Calculate insert_y to make total height multiple of 16
    m = math.ceil(seg_window_y / 2**conv_block_num)
    while 16 * m - seg_window_y < 2 * insert_xy:
        m = m + 1
    insert_y = int((16 * m - seg_window_y) / 2)
    
    # Ensure final height (seg_window_y + 2*insert_y) is multiple of 16
    total_height = seg_window_y + 2 * insert_y
    
    if total_height % 16 != 0:
        # Calculate target height (next multiple of 16)
        target_height = ((total_height // 16) + 1) * 16
        padding_needed = target_height - seg_window_y
        insert_y = padding_needed // 2
    
    # print(f"Segment window: {seg_window_x}x{seg_window_y}, Insert: {insert_x}x{insert_y}")
    
    # Calculate segment positions (exact replica)
    rr_list = list(range(0, inp_x - seg_window_x + 1, seg_window_x - overlap_x))
    if rr_list[-1] != inp_x - seg_window_x:
        rr_list.append(inp_x - seg_window_x)
    
    cc_list = list(range(0, inp_y - seg_window_y + 1, seg_window_y - overlap_y))
    if cc_list[-1] != inp_y - seg_window_y:
        cc_list.append(inp_y - seg_window_y)
    
    # print(f"Processing {len(rr_list)} x {len(cc_list)} segments...")
    
    # Segment (exact replica with padding for uniform size)
    segmented_inp = []
    original_sizes = []  # Store original sizes for each segment
    
    for rr in rr_list:
        for cc in cc_list:
            # Extract segment
            segment = slice_img[rr:rr + seg_window_x, cc:cc + seg_window_y]
            original_sizes.append(segment.shape)
            
            # Pad segment to uniform size if needed
            if segment.shape[0] < seg_window_x or segment.shape[1] < seg_window_y:
                pad_h = seg_window_x - segment.shape[0]
                pad_w = seg_window_y - segment.shape[1]
                segment = np.pad(segment, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            segmented_inp.append(segment)
    
    segmented_inp = np.array(segmented_inp).astype(np.float32)
    segmented_inp = segmented_inp[..., np.newaxis]
    seg_num = segmented_inp.shape[0]
    
    # Add padding to ensure multiple of 16
    # Calculate target sizes (multiples of 16)
    target_width = ((seg_window_x + 2 * insert_x + 15) // 16) * 16
    target_height = ((seg_window_y + 2 * insert_y + 15) // 16) * 16
    
    # Calculate exact padding needed
    width_padding_total = target_width - seg_window_x
    height_padding_total = target_height - seg_window_y
    
    # Distribute padding (left/right for width, top/bottom for height)
    width_pad_left = width_padding_total // 2
    width_pad_right = width_padding_total - width_pad_left
    height_pad_top = height_padding_total // 2
    height_pad_bottom = height_padding_total - height_pad_top
    
    # Create left and right padding for width
    insert_shape_left = np.zeros([seg_num, width_pad_left, seg_window_y, 1]).astype(np.float32)
    insert_shape_right = np.zeros([seg_num, width_pad_right, seg_window_y, 1]).astype(np.float32)
    
    # First concatenate along axis=1 (width)
    segmented_inp = np.concatenate((insert_shape_left, segmented_inp, insert_shape_right), axis=1)
    
    # Create top and bottom padding for height
    current_width = segmented_inp.shape[1]
    insert_shape_top = np.zeros([seg_num, current_width, height_pad_top, 1]).astype(np.float32)
    insert_shape_bottom = np.zeros([seg_num, current_width, height_pad_bottom, 1]).astype(np.float32)
    
    # Second concatenate along axis=2 (height)
    segmented_inp = np.concatenate((insert_shape_top, segmented_inp, insert_shape_bottom), axis=2)
    
    # Predict (process each segment)
    dec_list = np.zeros([seg_num, seg_window_x * (1 + upsample_flag), seg_window_y * (1 + upsample_flag)], dtype=np.float32)
    
    for seg_idx in range(seg_num):
        # Prepare input for model
        seg_input = segmented_inp[seg_idx]  # Shape: (H, W, 1)
        
        # Convert to tensor format expected by PyTorch model
        input_tensor = torch.from_numpy(seg_input).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, 1, H, W)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert back to numpy and remove padding
        result = output.squeeze().cpu().numpy()  # Shape: (H*2, W*2)
        
        # Crop result to remove padding (adjusted for new padding method)
        crop_left = width_pad_left * (1 + upsample_flag)
        crop_right = crop_left + seg_window_x * (1 + upsample_flag)
        crop_top = height_pad_top * (1 + upsample_flag)
        crop_bottom = crop_top + seg_window_y * (1 + upsample_flag)
        
        result_cropped = result[crop_left:crop_right, crop_top:crop_bottom]
        
        dec_list[seg_idx] = result_cropped
        
        # Clear GPU memory
        del input_tensor, output
        torch.cuda.empty_cache()
    
    # Fuse (exact replica of original logic)
    output_dec = np.zeros((inp_x * (1 + upsample_flag), inp_y * (1 + upsample_flag)), dtype=np.float32)
    
    # print('Fusing segments...')
    for r_ind, rr in enumerate(rr_list):
        for c_ind, cc in enumerate(cc_list):
            seg_idx = r_ind * len(cc_list) + c_ind
            original_h, original_w = original_sizes[seg_idx]
            
            # Calculate boundaries considering original segment size
            if rr == 0:
                rr_min = 0
                rr_min_patch = 0
            else:
                rr_min = rr + math.ceil(overlap_x / 2)
                rr_min_patch = math.ceil(overlap_x / 2)
            
            if rr + original_h == inp_x:
                rr_max = inp_x
                rr_max_patch = original_h
            else:
                rr_max = rr + original_h - math.floor(overlap_x / 2)
                rr_max_patch = original_h - math.floor(overlap_x / 2)
            
            if cc == 0:
                cc_min = 0
                cc_min_patch = 0
            else:
                cc_min = cc + math.ceil(overlap_y / 2)
                cc_min_patch = math.ceil(overlap_y / 2)
            
            if cc + original_w == inp_y:
                cc_max = inp_y
                cc_max_patch = original_w
            else:
                cc_max = cc + original_w - math.floor(overlap_y / 2)
                cc_max_patch = original_w - math.floor(overlap_y / 2)
            
            # Copy segment result to output (exact replica)
            cur_patch = dec_list[seg_idx,
                               rr_min_patch * (1 + upsample_flag):rr_max_patch * (1 + upsample_flag),
                               cc_min_patch * (1 + upsample_flag):cc_max_patch * (1 + upsample_flag)].astype(np.float32)
            
            output_dec[rr_min * (1 + upsample_flag):rr_max * (1 + upsample_flag),
                      cc_min * (1 + upsample_flag):cc_max * (1 + upsample_flag)] = cur_patch
    
    return output_dec


def process_slice_tiled(model, slice_img, tile_size, overlap, insert_xy, device, model_config=None):
    """Process a slice using tiling for large images - following original implementation."""
    import math
    
    height, width = slice_img.shape
    
    # Calculate tile positions (following original logic)
    step_size = tile_size - overlap
    
    # Calculate tile start positions
    h_positions = list(range(0, height - tile_size + 1, step_size))
    if h_positions[-1] != height - tile_size:
        h_positions.append(height - tile_size)
    
    w_positions = list(range(0, width - tile_size + 1, step_size))
    if w_positions[-1] != width - tile_size:
        w_positions.append(width - tile_size)
    
    # Initialize result (2x upsampling)
    result_height = height * 2
    result_width = width * 2
    result = np.zeros((result_height, result_width), dtype=np.float32)
    
    print(f"Processing {len(h_positions)} x {len(w_positions)} tiles...")
    
    # Process each tile
    for h_idx, h_start in enumerate(h_positions):
        for w_idx, w_start in enumerate(w_positions):
            # Extract tile
            tile = slice_img[h_start:h_start + tile_size, w_start:w_start + tile_size]
            
            # Process tile
            tile_result = process_slice_whole(model, tile, insert_xy, device)
            
            # Calculate the region to copy (following original fusing logic)
            # Determine the boundaries for this tile in the original image
            if h_start == 0:
                h_min = 0
                h_min_patch = 0
            else:
                h_min = h_start + math.ceil(overlap / 2)
                h_min_patch = math.ceil(overlap / 2)
            
            if h_start + tile_size == height:
                h_max = height
                h_max_patch = tile_size
            else:
                h_max = h_start + tile_size - math.floor(overlap / 2)
                h_max_patch = tile_size - math.floor(overlap / 2)
            
            if w_start == 0:
                w_min = 0
                w_min_patch = 0
            else:
                w_min = w_start + math.ceil(overlap / 2)
                w_min_patch = math.ceil(overlap / 2)
            
            if w_start + tile_size == width:
                w_max = width
                w_max_patch = tile_size
            else:
                w_max = w_start + tile_size - math.floor(overlap / 2)
                w_max_patch = tile_size - math.floor(overlap / 2)
            
            # Extract the non-overlapping part from the tile result (2x upsampling)
            patch_crop = tile_result[h_min_patch*2:h_max_patch*2, w_min_patch*2:w_max_patch*2]
            
            # Copy to result (direct assignment, no blending)
            result[h_min*2:h_max*2, w_min*2:w_max*2] = patch_crop
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained deconvolution model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--tile_size', type=int, default=None,
                       help='Tile size for processing large images (None = whole image)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles')
    parser.add_argument('--num_seg_window_x', type=int, default=4,
                       help='Number of segments in X direction (original method)')
    parser.add_argument('--num_seg_window_y', type=int, default=4,
                       help='Number of segments in Y direction (original method)')
    parser.add_argument('--overlap_x', type=int, default=20,
                       help='Overlap in X direction (original method)')
    parser.add_argument('--overlap_y', type=int, default=20,
                       help='Overlap in Y direction (original method)')
    parser.add_argument('--use_original_tiling', action='store_true',
                       help='Use original tiling method with num_seg_window parameters')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--pattern', type=str, default='*.tif',
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_config = load_model(args.checkpoint, device)
    
    # Find input images
    input_patterns = [args.pattern, args.pattern.replace('.tif', '.tiff')]
    input_files = []
    for pattern in input_patterns:
        input_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))
    
    input_files = sorted(list(set(input_files)))  # Remove duplicates
    
    if not input_files:
        print(f"No files found matching pattern '{args.pattern}' in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for input_file in input_files:
        try:
            process_single_image(
                model=model,
                model_config=model_config,
                image_path=input_file,
                output_dir=output_dir,
                device=device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                args=args
            )
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print("Inference completed!")


if __name__ == '__main__':
    main()