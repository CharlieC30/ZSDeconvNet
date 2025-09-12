#!/usr/bin/env python3
"""
ZS-DeconvNet 3D Inference Script - PyTorch Implementation  
Equivalent to original TensorFlow Infer_3D.py
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import glob
from typing import List, Tuple
import tifffile

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models import twostage_RCAN3D, twostage_Unet3D
from utils.utils import prctile_norm, read_tiff_stack, save_tiff_stack


class InferenceEngine:
    """
    Inference engine for ZS-DeconvNet 3D models
    Handles model loading, tiling, and inference
    """
    
    def __init__(self, 
                 model_path: str,
                 model_name: str = "twostage_RCAN3D",
                 device: str = 'auto',
                 upsample_flag: int = 0,
                 insert_xy: int = 8,
                 insert_z: int = 2):
        
        self.model_path = model_path
        self.model_name = model_name
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
        print(f"Loading model from: {self.model_path}")
        
        # Create model architecture
        dummy_input_shape = (80, 80, 17, 1)  # Will be adjusted based on actual input
        
        if self.model_name == "twostage_RCAN3D":
            model = twostage_RCAN3D.RCAN3D_factory(
                input_shape=dummy_input_shape,
                upsample_flag=self.upsample_flag,
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        elif self.model_name == "twostage_Unet3D":
            model = twostage_Unet3D.Unet(
                input_shape=dummy_input_shape,
                upsample_flag=self.upsample_flag,
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Load weights (updated for manual training approach)
        if self.model_path.endswith('.pth'):
            # PyTorch state dict from manual training
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # Our training format
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from iteration: {checkpoint.get('iteration', 'unknown')}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported model file format: {self.model_path}. Expected .pth file.")
        
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def infer_single(self, input_image: np.ndarray, background: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a single 3D image
        
        Args:
            input_image: Input image (Z, Y, X)
            background: Background value to subtract
            
        Returns:
            Tuple of (denoised, deconvolved) results
        """
        # Preprocess
        if background > 0:
            input_image = input_image - background
            input_image = np.clip(input_image, 0, None)
        
        # Normalize
        input_image = input_image.astype(np.float32)
        input_image = input_image / np.max(input_image) if np.max(input_image) > 0 else input_image
        
        # Add padding (manual implementation if pad_3d not available)
        padded = np.pad(input_image, 
                       ((self.insert_z, self.insert_z), 
                        (self.insert_xy, self.insert_xy), 
                        (self.insert_xy, self.insert_xy)), 
                       mode='constant', constant_values=0)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            denoise_out, deconv_out = self.model(input_tensor)
        
        # Convert back to numpy
        denoise_np = denoise_out.squeeze().cpu().numpy()
        deconv_np = deconv_out.squeeze().cpu().numpy()
        
        # Remove padding and crop to original size
        original_shape = input_image.shape
        
        # For denoising output (no upsampling)
        denoise_cropped = denoise_np[
            self.insert_z:self.insert_z + original_shape[0],
            self.insert_xy:self.insert_xy + original_shape[1], 
            self.insert_xy:self.insert_xy + original_shape[2]
        ]
        
        # For deconvolution output (may have upsampling)
        if self.upsample_flag:
            crop_xy = self.insert_xy * 2
            deconv_cropped = deconv_np[
                self.insert_z:self.insert_z + original_shape[0],
                crop_xy:crop_xy + original_shape[1] * 2,
                crop_xy:crop_xy + original_shape[2] * 2
            ]
        else:
            deconv_cropped = deconv_np[
                self.insert_z:self.insert_z + original_shape[0],
                self.insert_xy:self.insert_xy + original_shape[1],
                self.insert_xy:self.insert_xy + original_shape[2]
            ]
        
        return denoise_cropped, deconv_cropped
    
    def infer_with_tiling(self, 
                         input_image: np.ndarray,
                         tile_size: Tuple[int, int, int] = (64, 64, 13),
                         overlap: Tuple[int, int, int] = (8, 8, 2),
                         background: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference with tiling for large images
        
        Args:
            input_image: Input image (Z, Y, X)
            tile_size: Size of each tile (Z, Y, X) 
            overlap: Overlap between tiles (Z, Y, X)
            background: Background value
            
        Returns:
            Tuple of (denoised, deconvolved) results
        """
        z, y, x = input_image.shape
        tz, ty, tx = tile_size
        oz, oy, ox = overlap
        
        # Calculate number of tiles
        n_tiles_z = max(1, (z - oz) // (tz - oz) + (1 if (z - oz) % (tz - oz) > 0 else 0))
        n_tiles_y = max(1, (y - oy) // (ty - oy) + (1 if (y - oy) % (ty - oy) > 0 else 0))
        n_tiles_x = max(1, (x - ox) // (tx - ox) + (1 if (x - ox) % (tx - ox) > 0 else 0))
        
        print(f"Processing {n_tiles_z}x{n_tiles_y}x{n_tiles_x} tiles")
        
        # Initialize output arrays
        scale_factor = 2 if self.upsample_flag else 1
        denoise_result = np.zeros_like(input_image)
        deconv_result = np.zeros((z, y * scale_factor, x * scale_factor))
        weight_map = np.zeros_like(input_image)
        
        for iz in range(n_tiles_z):
            for iy in range(n_tiles_y):
                for ix in range(n_tiles_x):
                    # Calculate tile boundaries
                    z_start = iz * (tz - oz)
                    z_end = min(z_start + tz, z)
                    y_start = iy * (ty - oy)
                    y_end = min(y_start + ty, y)
                    x_start = ix * (tx - ox)
                    x_end = min(x_start + tx, x)
                    
                    # Extract tile
                    tile = input_image[z_start:z_end, y_start:y_end, x_start:x_end]
                    
                    # Process tile
                    denoise_tile, deconv_tile = self.infer_single(tile, background)
                    
                    # Blend results (simple averaging in overlap regions)
                    denoise_result[z_start:z_end, y_start:y_end, x_start:x_end] += denoise_tile
                    deconv_result[z_start:z_end, 
                                y_start*scale_factor:y_end*scale_factor,
                                x_start*scale_factor:x_end*scale_factor] += deconv_tile
                    weight_map[z_start:z_end, y_start:y_end, x_start:x_end] += 1
        
        # Normalize by weight map
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        denoise_result = denoise_result / weight_map
        
        # For deconv result, create corresponding weight map
        deconv_weight = np.zeros_like(deconv_result)
        deconv_weight[:, ::scale_factor, ::scale_factor] = weight_map
        if scale_factor > 1:
            deconv_weight = F.interpolate(
                torch.from_numpy(deconv_weight).unsqueeze(0).unsqueeze(0),
                scale_factor=(1, scale_factor, scale_factor),
                mode='trilinear'
            ).squeeze().numpy()
        deconv_weight[deconv_weight == 0] = 1
        deconv_result = deconv_result / deconv_weight
        
        return denoise_result, deconv_result


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='ZS-DeconvNet 3D Inference')
    
    # Model parameters
    parser.add_argument("--model", type=str, default="twostage_RCAN3D")
    parser.add_argument("--load_weights_path", type=str, required=True)
    parser.add_argument("--upsample_flag", type=int, default=0)
    parser.add_argument("--insert_xy", type=int, default=8)
    parser.add_argument("--insert_z", type=int, default=2)
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--background", type=int, default=100)
    
    # Tiling parameters
    parser.add_argument("--use_tiling", action='store_true')
    parser.add_argument("--tile_size_z", type=int, default=64)
    parser.add_argument("--tile_size_y", type=int, default=64) 
    parser.add_argument("--tile_size_x", type=int, default=64)
    parser.add_argument("--overlap_z", type=int, default=8)
    parser.add_argument("--overlap_y", type=int, default=8)
    parser.add_argument("--overlap_x", type=int, default=8)
    
    # Device
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        model_dir = os.path.dirname(args.load_weights_path)
        args.output_dir = os.path.join(model_dir, 'Inference')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=args.load_weights_path,
        model_name=args.model,
        device=args.device,
        upsample_flag=args.upsample_flag,
        insert_xy=args.insert_xy,
        insert_z=args.insert_z
    )
    
    # Find input files
    if os.path.isfile(args.input_dir):
        input_files = [args.input_dir]
    else:
        extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        input_files = []
        for ext in extensions:
            input_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        
        if not input_files:
            print(f"No image files found in {args.input_dir}")
            return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for i, file_path in enumerate(input_files):
        print(f"\nProcessing [{i+1}/{len(input_files)}]: {os.path.basename(file_path)}")
        
        # Load image
        try:
            image = read_tiff_stack(file_path)
            print(f"Image shape: {image.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        # Perform inference
        try:
            if args.use_tiling:
                tile_size = (args.tile_size_z, args.tile_size_y, args.tile_size_x)
                overlap = (args.overlap_z, args.overlap_y, args.overlap_x)
                denoise_result, deconv_result = engine.infer_with_tiling(
                    image, tile_size, overlap, args.background
                )
            else:
                denoise_result, deconv_result = engine.infer_single(image, args.background)
            
            print(f"Denoised shape: {denoise_result.shape}")
            print(f"Deconvolved shape: {deconv_result.shape}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
        
        # Save results
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save denoised result
        denoise_path = os.path.join(args.output_dir, f"{base_name}_denoised.tif")
        denoise_scaled = prctile_norm(denoise_result)
        save_tiff_stack(denoise_path, denoise_scaled, dtype='uint16')
        print(f"Saved denoised: {denoise_path}")
        
        # Save deconvolved result  
        deconv_path = os.path.join(args.output_dir, f"{base_name}_deconvolved.tif")
        deconv_scaled = prctile_norm(deconv_result)
        save_tiff_stack(deconv_path, deconv_scaled, dtype='uint16')
        print(f"Saved deconvolved: {deconv_path}")
    
    print(f"\nInference completed! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()