#!/usr/bin/env python3
"""
Debug script to trace channel flow through the model.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.deconv_unet import DeconvUNet

def debug_forward_pass():
    """Debug the forward pass to see channel dimensions."""
    
    config = {
        'input_channels': 1,
        'output_channels': 1,
        'conv_block_num': 4,
        'conv_num': 3,
        'upsample_flag': True,
        'insert_xy': 16
    }
    
    model = DeconvUNet(**config)
    model.eval()
    
    x = torch.randn(1, 1, 160, 160)
    
    print(f"Input shape: {x.shape}")
    
    # Forward through encoder
    encoder_outputs = []
    current = x
    
    for i, encoder_block in enumerate(model.encoder_blocks):
        current, conv_out = encoder_block(current)
        encoder_outputs.append(conv_out)
        print(f"Encoder block {i}: pooled={current.shape}, conv_out={conv_out.shape}")
    
    # Middle layers
    current = model.mid_relu(model.mid_conv1(current))
    current = model.mid_relu(model.mid_conv2(current))
    print(f"After middle: {current.shape}")
    
    # Debug decoder
    for i, decoder_block in enumerate(model.decoder_blocks):
        skip_connection = encoder_outputs[-(i+1)]
        print(f"\nDecoder block {i}:")
        print(f"  Current (before upsample): {current.shape}")
        
        # Manual upsample
        current_up = decoder_block.upsample(current)
        print(f"  Current (after upsample): {current_up.shape}")
        print(f"  Skip connection: {skip_connection.shape}")
        
        # Concatenate
        concat = torch.cat([current_up, skip_connection], dim=1)
        print(f"  After concatenation: {concat.shape}")
        
        # Check if this matches expected input channels for decoder block
        expected_channels = decoder_block.conv_block[0].in_channels
        print(f"  Expected channels for decoder block: {expected_channels}")
        
        if concat.shape[1] != expected_channels:
            print(f"  ❌ MISMATCH! Got {concat.shape[1]}, expected {expected_channels}")
            return False
        else:
            print(f"  ✅ Channels match!")
        
        # Continue forward
        current = decoder_block.conv_block(concat)
        print(f"  Output: {current.shape}")
    
    print(f"\nFinal processing...")
    if model.upsample_flag:
        current = model.final_upsample(current)
        print(f"After final upsample: {current.shape}")
    
    current = model.final_relu(model.final_conv1(current))
    current = model.final_relu(model.final_conv2(current))
    output = model.final_relu(model.final_conv3(current))
    
    print(f"Final output: {output.shape}")
    return True

if __name__ == "__main__":
    debug_forward_pass()