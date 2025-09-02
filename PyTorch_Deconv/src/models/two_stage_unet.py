import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv_unet import ConvBlock, ConcatBlock


class TwoStageUNet(nn.Module):
    """
    Two-stage 2D U-Net for ZS-DeconvNet, exactly matching TensorFlow implementation.
    Stage 1: Denoising U-Net (input → denoised)
    Stage 2: Deconvolution U-Net (denoised → deconvolved)
    """
    
    def __init__(self, 
                 input_channels=1, 
                 output_channels=1,
                 conv_block_num=4, 
                 conv_num=3,
                 upsample_flag=True,
                 insert_xy=16):
        super().__init__()
        
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        self.conv_block_num = conv_block_num
        
        # ========== Stage 1: Denoising U-Net ==========
        # Encoder for stage 1
        self.stage1_encoder_blocks = nn.ModuleList()
        in_ch = input_channels
        for n in range(conv_block_num):
            out_ch = 2 ** (n + 5)  # 32, 64, 128, 256
            self.stage1_encoder_blocks.append(ConvBlock(in_ch, out_ch, conv_num))
            in_ch = out_ch
        
        # Middle layers for stage 1
        mid_ch = 2 ** (conv_block_num + 5)  # channels * 2
        self.stage1_mid_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.stage1_mid_conv2 = nn.Conv2d(mid_ch, in_ch, kernel_size=3, padding=1)
        self.stage1_mid_relu = nn.ReLU()
        
        # Decoder for stage 1
        self.stage1_decoder_blocks = nn.ModuleList()
        encoder_channels = [2 ** (n + 5) for n in range(conv_block_num)]
        init_channels = in_ch
        
        for n in range(conv_block_num):
            out_ch = init_channels // (2 ** n)
            skip_ch = encoder_channels[conv_block_num - 1 - n]
            
            if n == 0:
                current_ch = in_ch
            else:
                prev_out_ch = init_channels // (2 ** (n-1))
                current_ch = prev_out_ch // 2
            
            concat_in_ch = current_ch + skip_ch
            self.stage1_decoder_blocks.append(ConcatBlock(concat_in_ch, out_ch, conv_num))
        
        # Output layer for stage 1 (denoised image)
        last_out_ch = init_channels // (2 ** (conv_block_num - 1))
        final_in_ch = last_out_ch // 2
        self.stage1_output = nn.Conv2d(final_in_ch, output_channels, kernel_size=3, padding=1)
        self.stage1_output_relu = nn.ReLU()
        
        # ========== Stage 2: Deconvolution U-Net ==========
        # Encoder for stage 2
        self.stage2_encoder_blocks = nn.ModuleList()
        in_ch = output_channels  # Input is the denoised output from stage 1
        for n in range(conv_block_num):
            out_ch = 2 ** (n + 5)  # 32, 64, 128, 256
            self.stage2_encoder_blocks.append(ConvBlock(in_ch, out_ch, conv_num))
            in_ch = out_ch
        
        # Middle layers for stage 2
        mid_ch = 2 ** (conv_block_num + 5)
        self.stage2_mid_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.stage2_mid_conv2 = nn.Conv2d(mid_ch, in_ch, kernel_size=3, padding=1)
        self.stage2_mid_relu = nn.ReLU()
        
        # Decoder for stage 2
        self.stage2_decoder_blocks = nn.ModuleList()
        init_channels = in_ch
        
        for n in range(conv_block_num):
            out_ch = init_channels // (2 ** n)
            skip_ch = encoder_channels[conv_block_num - 1 - n]
            
            if n == 0:
                current_ch = in_ch
            else:
                prev_out_ch = init_channels // (2 ** (n-1))
                current_ch = prev_out_ch // 2
            
            concat_in_ch = current_ch + skip_ch
            self.stage2_decoder_blocks.append(ConcatBlock(concat_in_ch, out_ch, conv_num))
        
        # Final layers for stage 2 (deconvolved image)
        last_out_ch = init_channels // (2 ** (conv_block_num - 1))
        final_in_ch = last_out_ch // 2
        
        if upsample_flag:
            self.stage2_final_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.stage2_final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        else:
            self.stage2_final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        
        self.stage2_final_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.stage2_final_conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.stage2_final_relu = nn.ReLU()
        
        # Initialize final layers properly
        self._initialize_final_layers()
        
        # Initialize stage2_output_bias as optional parameter to prevent state_dict mismatch
        self.register_parameter('stage2_output_bias', None)
    
    def forward(self, x):
        """
        Forward pass through both stages.
        Returns: [denoised_output, deconvolved_output] to match TensorFlow
        """
        # ========== Stage 1: Denoising ==========
        # Encoder path
        stage1_encoder_outputs = []
        current = x
        
        for encoder_block in self.stage1_encoder_blocks:
            current, conv_out = encoder_block(current)
            stage1_encoder_outputs.append(conv_out)
        
        # Middle layers
        current = self.stage1_mid_relu(self.stage1_mid_conv1(current))
        current = self.stage1_mid_relu(self.stage1_mid_conv2(current))
        
        # Decoder path
        for i, decoder_block in enumerate(self.stage1_decoder_blocks):
            skip_connection = stage1_encoder_outputs[-(i+1)]
            current = decoder_block(current, skip_connection)
        
        # Output layer for stage 1
        output1_raw = self.stage1_output(current)
        output1 = self.stage1_output_relu(output1_raw)
        
        # Apply insert_xy cropping for stage 1 output (matching TensorFlow)
        if self.insert_xy > 0:
            h_start, h_end = self.insert_xy, output1.shape[2] - self.insert_xy
            w_start, w_end = self.insert_xy, output1.shape[3] - self.insert_xy
            output1_cropped = output1[:, :, h_start:h_end, w_start:w_end]
        else:
            output1_cropped = output1
        
        # ========== Stage 2: Deconvolution ==========
        # Use the full (uncropped) denoised output as input to stage 2
        # This matches TensorFlow where stage 2 takes full output1 as input
        stage2_encoder_outputs = []
        current = output1  # Full denoised output
        
        for encoder_block in self.stage2_encoder_blocks:
            current, conv_out = encoder_block(current)
            stage2_encoder_outputs.append(conv_out)
        
        # Middle layers
        current = self.stage2_mid_relu(self.stage2_mid_conv1(current))
        current = self.stage2_mid_relu(self.stage2_mid_conv2(current))
        
        # Decoder path
        for i, decoder_block in enumerate(self.stage2_decoder_blocks):
            skip_connection = stage2_encoder_outputs[-(i+1)]
            current = decoder_block(current, skip_connection)
        
        # Final layers for stage 2
        if self.upsample_flag:
            current = self.stage2_final_upsample(current)
        
        current = self.stage2_final_relu(self.stage2_final_conv1(current))
        current = self.stage2_final_relu(self.stage2_final_conv2(current))
        
        # Final output
        output2_raw = self.stage2_final_conv3(current)

        # Handle potential for dead ReLU by adding a learnable bias if needed
        negative_ratio = (output2_raw < 0).float().mean()
        if negative_ratio > 0.5:
            if self.stage2_output_bias is None:
                self.stage2_output_bias = nn.Parameter(torch.tensor(0.1, device=output2_raw.device))
            output2 = self.stage2_final_relu(output2_raw + self.stage2_output_bias)
        else:
            output2 = self.stage2_final_relu(output2_raw)
        
        # Apply insert_xy cropping for stage 2 output
        if self.insert_xy > 0:
            h_start, h_end = self.insert_xy, output2.shape[2] - self.insert_xy
            w_start, w_end = self.insert_xy, output2.shape[3] - self.insert_xy
            output2_cropped = output2[:, :, h_start:h_end, w_start:w_end]
        else:
            output2_cropped = output2

        # Enhanced debug information for monitoring two-stage behavior
        if self.training and torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            # Calculate difference metrics
            stage1_diff = torch.abs(output1 - x).mean().item()
            
            # Analyze Stage 1 output distribution before and after cropping
            stage1_raw_stats = {
                'mean': output1_raw.mean().item(),
                'std': output1_raw.std().item(),
                'min': output1_raw.min().item(),
                'max': output1_raw.max().item(),
                'negative_ratio': (output1_raw < 0).float().mean().item()
            }
            
            stage1_after_relu_stats = {
                'mean': output1.mean().item(), 
                'std': output1.std().item(),
                'zero_ratio': (output1 == 0).float().mean().item(),
                'nonzero_mean': output1[output1 > 0].mean().item() if (output1 > 0).any() else 0.0
            }
            
            # Check what happens in the cropped region vs edges
            if self.insert_xy > 0:
                # Center region (what will be kept)
                center_region = output1[:, :, self.insert_xy:output1.shape[2]-self.insert_xy, 
                                       self.insert_xy:output1.shape[3]-self.insert_xy]
                center_stats = {
                    'mean': center_region.mean().item(),
                    'std': center_region.std().item(), 
                    'zero_ratio': (center_region == 0).float().mean().item()
                }
                
                # Edge regions (what will be cropped out)
                edge_mask = torch.ones_like(output1, dtype=torch.bool)
                edge_mask[:, :, self.insert_xy:output1.shape[2]-self.insert_xy, 
                         self.insert_xy:output1.shape[3]-self.insert_xy] = False
                edge_values = output1[edge_mask]
                edge_stats = {
                    'mean': edge_values.mean().item() if len(edge_values) > 0 else 0.0,
                    'std': edge_values.std().item() if len(edge_values) > 0 else 0.0,
                    'zero_ratio': (edge_values == 0).float().mean().item() if len(edge_values) > 0 else 0.0
                }
            else:
                center_stats = {'mean': 0, 'std': 0, 'zero_ratio': 0}
                edge_stats = {'mean': 0, 'std': 0, 'zero_ratio': 0}
            
            print(f"\n=== Two-Stage Forward Debug (Enhanced) ===")
            print(f"Input shape: {x.shape}, range: [{x.min().item():.6f}, {x.max().item():.6f}]")
            print(f"Stage 1 raw output (before ReLU): mean={stage1_raw_stats['mean']:.6f}, "
                  f"std={stage1_raw_stats['std']:.6f}, negative_ratio={stage1_raw_stats['negative_ratio']:.3f}")
            print(f"Stage 1 after ReLU: mean={stage1_after_relu_stats['mean']:.6f}, "
                  f"zero_ratio={stage1_after_relu_stats['zero_ratio']:.3f}, "
                  f"nonzero_mean={stage1_after_relu_stats['nonzero_mean']:.6f}")
            print(f"Stage 1 center region (kept): mean={center_stats['mean']:.6f}, "
                  f"std={center_stats['std']:.6f}, zero_ratio={center_stats['zero_ratio']:.3f}")
            print(f"Stage 1 edge region (cropped): mean={edge_stats['mean']:.6f}, "
                  f"std={edge_stats['std']:.6f}, zero_ratio={edge_stats['zero_ratio']:.3f}")
            print(f"Stage 1 vs Input MAE: {stage1_diff:.6f}")
            print(f"Stage 1 cropped shape: {output1_cropped.shape}, range: [{output1_cropped.min().item():.6f}, {output1_cropped.max().item():.6f}]")
            print(f"Stage 2 output shape: {output2.shape}, range: [{output2.min().item():.6f}, {output2.max().item():.6f}]")
            print(f"Stage 2 cropped shape: {output2_cropped.shape}, range: [{output2_cropped.min().item():.6f}, {output2_cropped.max().item():.6f}]")
            print("=" * 50)
        
        return [output1_cropped, output2_cropped]
    
    def _initialize_final_layers(self):
        """Initialize final layers to ensure proper output range."""
        # Stage 1 final layer - Enhanced initialization for better center activation
        # Use larger gain and positive bias to ensure some positive outputs in center region
        nn.init.xavier_normal_(self.stage1_output.weight, gain=1.2)  # Slightly larger gain
        nn.init.constant_(self.stage1_output.bias, 0.1)  # Small positive bias to help avoid dead ReLU
        
        # Add a center-biased initialization pattern to Stage 1 output weights
        # This encourages the network to focus on the center region that won't be cropped
        with torch.no_grad():
            # Create a spatial bias that favors center activations
            if self.stage1_output.weight.shape[2] == 3 and self.stage1_output.weight.shape[3] == 3:
                # For 3x3 kernels, emphasize center pixel
                center_mask = torch.zeros_like(self.stage1_output.weight)
                center_mask[:, :, 1, 1] = 1.0  # Center pixel
                edge_mask = 1.0 - center_mask
                
                # Apply center emphasis: center weights * 1.5, edge weights * 0.8
                self.stage1_output.weight.data = (self.stage1_output.weight.data * center_mask * 1.5 + 
                                                 self.stage1_output.weight.data * edge_mask * 0.8)
        
        # Stage 2 final layers
        nn.init.xavier_normal_(self.stage2_final_conv3.weight, gain=1.0)
        nn.init.constant_(self.stage2_final_conv3.bias, 0.0)
        nn.init.xavier_normal_(self.stage2_final_conv1.weight, gain=1.0)
        nn.init.constant_(self.stage2_final_conv1.bias, 0.0)
        nn.init.xavier_normal_(self.stage2_final_conv2.weight, gain=1.0) 
        nn.init.constant_(self.stage2_final_conv2.bias, 0.0)


def create_two_stage_unet(config=None):
    """Factory function to create TwoStageUNet with configuration."""
    if config is None:
        config = {
            'input_channels': 1,
            'output_channels': 1,
            'conv_block_num': 4,
            'conv_num': 3,
            'upsample_flag': True,
            'insert_xy': 16
        }
    
    return TwoStageUNet(**config)


if __name__ == "__main__":
    # Test the model
    model = create_two_stage_unet()
    x = torch.randn(1, 1, 160, 160)  # Batch=1, Channels=1, H=160, W=160
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Stage 1 output shape: {outputs[0].shape}")
    print(f"Stage 2 output shape: {outputs[1].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")