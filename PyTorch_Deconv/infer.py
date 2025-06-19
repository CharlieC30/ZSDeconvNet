#!/usr/bin/env python3
"""
Inference script for PyTorch Lightning deconvolution model.
"""

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
from src.data.datamodule import InferenceDataset
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
    
    # For inference, we only need the U-Net model, not the Lightning module
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
    
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    
    return model, model_config


def process_single_image(model, model_config, image_path, output_dir, device='cpu', 
                        tile_size=None, overlap=64, batch_size=1):
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
    
    # Normalize
    img = prctile_norm(img)
    
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
        
        if tile_size is None or (height <= tile_size and width <= tile_size):
            # Process whole slice
            processed_slice = process_slice_whole(model, slice_img, insert_xy, device)
        else:
            # Process with tiling
            processed_slice = process_slice_tiled(
                model, slice_img, tile_size, overlap, insert_xy, device
            )
        
        processed_slices.append(processed_slice)
    
    # Stack processed slices
    if is_2d:
        result = processed_slices[0]
    else:
        result = np.stack(processed_slices, axis=0)
    
    # Save result
    output_path = output_dir / f"{Path(image_path).stem}_deconvolved.tif"
    
    # Convert to uint16 for saving
    result_uint16 = (prctile_norm(result) * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, result_uint16)
    
    print(f"Saved result to: {output_path}")
    return output_path


def process_slice_whole(model, slice_img, insert_xy, device):
    """Process a single slice without tiling."""
    # Normalize input like in training (percentile normalization)
    slice_norm = (slice_img - np.percentile(slice_img, 0)) / (np.percentile(slice_img, 100) - np.percentile(slice_img, 0) + 1e-7)
    slice_norm = np.clip(slice_norm, 0, 1)
    
    # Add padding
    pad_width = ((insert_xy, insert_xy), (insert_xy, insert_xy))
    padded_img = np.pad(slice_norm, pad_width, mode='constant', constant_values=0)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert back to numpy and crop to original scale
    result = output.squeeze().cpu().numpy()
    
    # Calculate crop coordinates to get 2x original size
    original_h, original_w = slice_img.shape
    expected_h, expected_w = original_h * 2, original_w * 2
    
    result_h, result_w = result.shape
    crop_h = (result_h - expected_h) // 2
    crop_w = (result_w - expected_w) // 2
    
    if crop_h > 0 or crop_w > 0:
        result = result[crop_h:crop_h + expected_h, crop_w:crop_w + expected_w]
    
    return result


def process_slice_tiled(model, slice_img, tile_size, overlap, insert_xy, device):
    """Process a slice using tiling for large images."""
    height, width = slice_img.shape
    
    # Calculate tile positions
    step_size = tile_size - overlap
    h_tiles = (height - overlap) // step_size + (1 if (height - overlap) % step_size else 0)
    w_tiles = (width - overlap) // step_size + (1 if (width - overlap) % step_size else 0)
    
    # Initialize result
    if model_config.get('upsample_flag', True):
        result_height = height * 2
        result_width = width * 2
    else:
        result_height = height
        result_width = width
    
    result = np.zeros((result_height, result_width), dtype=np.float32)
    weight_map = np.zeros((result_height, result_width), dtype=np.float32)
    
    # Process tiles
    for h_idx in range(h_tiles):
        for w_idx in range(w_tiles):
            # Calculate tile boundaries
            h_start = h_idx * step_size
            h_end = min(h_start + tile_size, height)
            w_start = w_idx * step_size
            w_end = min(w_start + tile_size, width)
            
            # Extract tile
            tile = slice_img[h_start:h_end, w_start:w_end]
            
            # Pad tile to tile_size if needed
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            # Process tile
            tile_result = process_slice_whole(model, tile, insert_xy, device)
            
            # Remove padding from result if added
            if pad_h > 0:
                tile_result = tile_result[:-pad_h]
            if pad_w > 0:
                tile_result = tile_result[:, :-pad_w]
            
            # Calculate result coordinates
            if model_config.get('upsample_flag', True):
                result_h_start = h_start * 2
                result_h_end = h_end * 2
                result_w_start = w_start * 2
                result_w_end = w_end * 2
            else:
                result_h_start = h_start
                result_h_end = h_end
                result_w_start = w_start
                result_w_end = w_end
            
            # Add to result with blending
            tile_height, tile_width = tile_result.shape
            result[result_h_start:result_h_start + tile_height, 
                   result_w_start:result_w_start + tile_width] += tile_result
            weight_map[result_h_start:result_h_start + tile_height, 
                      result_w_start:result_w_start + tile_width] += 1
    
    # Normalize by weight map
    result = np.divide(result, weight_map, out=np.zeros_like(result), where=weight_map!=0)
    
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
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print("Inference completed!")


if __name__ == '__main__':
    main()