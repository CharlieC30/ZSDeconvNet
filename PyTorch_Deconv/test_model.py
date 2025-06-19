#!/usr/bin/env python3
"""
Test script to verify model architecture and forward pass.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.deconv_unet import DeconvUNet

def test_model_forward():
    """Test the model forward pass with different input sizes."""
    
    print("Testing DeconvUNet architecture...")
    
    # Test configurations
    configs = [
        {
            'input_channels': 1,
            'output_channels': 1,
            'conv_block_num': 4,
            'conv_num': 3,
            'upsample_flag': True,
            'insert_xy': 16
        }
    ]
    
    test_inputs = [
        (1, 1, 160, 160),  # (batch, channels, height, width)
        (2, 1, 192, 192),
        (1, 1, 128, 128),
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        try:
            model = DeconvUNet(**config)
            print(f"✓ Model created successfully")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            model.eval()
            
            for j, input_shape in enumerate(test_inputs):
                print(f"\n  Test input {j+1}: {input_shape}")
                
                try:
                    with torch.no_grad():
                        x = torch.randn(*input_shape)
                        output = model(x)
                        print(f"  ✓ Forward pass successful")
                        print(f"    Output shape: {output.shape}")
                        
                        # Check output is reasonable
                        if torch.isnan(output).any():
                            print(f"  ⚠ Output contains NaN values")
                        elif torch.isinf(output).any():
                            print(f"  ⚠ Output contains infinite values")
                        else:
                            print(f"    Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                        
                except Exception as e:
                    print(f"  ✗ Forward pass failed: {e}")
                    
                    # Print model structure for debugging
                    print("\n  Model structure:")
                    for name, module in model.named_modules():
                        if len(list(module.children())) == 0:  # Leaf modules only
                            print(f"    {name}: {module}")
                    
                    return False
                    
        except Exception as e:
            print(f"✗ Model creation failed: {e}")
            return False
    
    print(f"\n✅ All tests passed!")
    return True

def print_model_info():
    """Print detailed model information."""
    
    config = {
        'input_channels': 1,
        'output_channels': 1,
        'conv_block_num': 4,
        'conv_num': 3,
        'upsample_flag': True,
        'insert_xy': 16
    }
    
    model = DeconvUNet(**config)
    
    print("\n" + "="*50)
    print("DETAILED MODEL ARCHITECTURE")
    print("="*50)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
        print(f"{name:40} {str(param.shape):20} {params:>10,}")
    
    print("-" * 50)
    print(f"{'Total parameters:':40} {total_params:>20,}")
    print(f"{'Trainable parameters:':40} {trainable_params:>20,}")
    print(f"{'Non-trainable parameters:':40} {total_params - trainable_params:>20,}")
    print(f"{'Model size (MB):':40} {total_params * 4 / (1024**2):>20.2f}")

if __name__ == "__main__":
    success = test_model_forward()
    
    if success:
        print_model_info()
    else:
        print("\n❌ Model test failed. Please check the architecture.")
        sys.exit(1)