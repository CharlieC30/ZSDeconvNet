#!/usr/bin/env python3
"""
Debug inference to check model output values.
"""

import torch
import numpy as np
import tifffile
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.deconv_unet import DeconvUNet
from src.utils.psf_utils import prctile_norm

def debug_model_output():
    """Debug model output values step by step."""
    
    # Load model configuration (same as inference)
    model_config = {
        'input_channels': 1,
        'output_channels': 1,
        'conv_block_num': 4,
        'conv_num': 3,
        'upsample_flag': True,
        'insert_xy': 16
    }
    
    # Create model
    model = DeconvUNet(**model_config)
    
    # Load checkpoint
    checkpoint_path = 'PyTorch_Deconv/Data/Output/final_model.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model weights
    state_dict = checkpoint['state_dict']
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            model_state_dict[new_key] = value
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Load a sample input
    input_img = tifffile.imread('PyTorch_Deconv/Data/Inference/DPM4Xsample_2070.tif')
    print(f"Input image shape: {input_img.shape}")
    print(f"Input image dtype: {input_img.dtype}")
    print(f"Input image range: [{input_img.min()}, {input_img.max()}]")
    
    # Take one slice for testing
    slice_img = input_img[0].astype(np.float32)
    print(f"Slice shape: {slice_img.shape}")
    print(f"Slice range: [{slice_img.min()}, {slice_img.max()}]")
    
    # Normalize slice like in the dataset
    slice_norm = (slice_img - np.percentile(slice_img, 0)) / (np.percentile(slice_img, 100) - np.percentile(slice_img, 0) + 1e-7)
    slice_norm = np.clip(slice_norm, 0, 1)
    print(f"Normalized slice range: [{slice_norm.min()}, {slice_norm.max()}]")
    
    # Add padding
    insert_xy = 16
    pad_width = ((insert_xy, insert_xy), (insert_xy, insert_xy))
    padded_img = np.pad(slice_norm, pad_width, mode='constant', constant_values=0)
    print(f"Padded image shape: {padded_img.shape}")
    
    # Convert to tensor
    input_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor range: [{input_tensor.min()}, {input_tensor.max()}]")
    
    # Run model
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output range: [{output.min()}, {output.max()}]")
    print(f"Model output mean: {output.mean()}")
    print(f"Model output std: {output.std()}")
    
    # Check if output has NaN or inf
    if torch.isnan(output).any():
        print("WARNING: Output contains NaN values!")
    if torch.isinf(output).any():
        print("WARNING: Output contains infinite values!")
    
    # Convert to numpy
    output_np = output.squeeze().cpu().numpy()
    print(f"Output numpy shape: {output_np.shape}")
    print(f"Output numpy range: [{output_np.min()}, {output_np.max()}]")
    
    # Test percentile normalization
    if output_np.max() > output_np.min():  # Check for non-constant output
        norm_output = prctile_norm(output_np)
        print(f"Normalized output range: [{norm_output.min()}, {norm_output.max()}]")
        
        # Convert to uint16
        uint16_output = (norm_output * 65535).astype(np.uint16)
        print(f"Final uint16 range: [{uint16_output.min()}, {uint16_output.max()}]")
    else:
        print("ERROR: Model output is constant (all same values)!")
        print("This explains why the saved image is all black.")
    
    # Test with raw input (no normalization)
    print("\n--- Testing with raw input (no normalization) ---")
    
    # Scale raw input to [0, 1] range
    raw_slice = slice_img / slice_img.max()
    padded_raw = np.pad(raw_slice, pad_width, mode='constant', constant_values=0)
    raw_tensor = torch.from_numpy(padded_raw).unsqueeze(0).unsqueeze(0)
    
    print(f"Raw input tensor range: [{raw_tensor.min()}, {raw_tensor.max()}]")
    
    with torch.no_grad():
        raw_output = model(raw_tensor)
    
    print(f"Raw output range: [{raw_output.min()}, {raw_output.max()}]")
    print(f"Raw output mean: {raw_output.mean()}")

if __name__ == "__main__":
    debug_model_output()