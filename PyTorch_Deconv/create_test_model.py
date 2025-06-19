#!/usr/bin/env python3
"""
Create a test model for inference testing.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.deconv_unet import DeconvUNet

def create_test_model():
    """Create a test model with random weights for inference testing."""
    
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
    
    # Initialize with small random weights (better than zeros)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param, gain=0.1)
        else:
            torch.nn.init.zeros_(param)
    
    # Create a mock checkpoint with the model state dict
    checkpoint = {
        'state_dict': {f'model.{k}': v for k, v in model.state_dict().items()},
        'hyper_parameters': {
            'model_config': model_config
        }
    }
    
    # Save the test model
    output_path = 'PyTorch_Deconv/Data/Output/test_model.ckpt'
    torch.save(checkpoint, output_path)
    print(f"Test model saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_test_model()