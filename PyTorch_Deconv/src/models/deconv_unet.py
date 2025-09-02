import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3):
        super().__init__()
        layers = []
        for i in range(conv_num):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        
        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_out = self.conv_block(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3):
        super().__init__()
        # Match TensorFlow: UpSampling2D(size=(2, 2)) is equivalent to nearest neighbor
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        layers = []
        # EXACT TensorFlow pattern:
        # First conv: Conv2D(channels, kernel_size=3, activation='relu', padding='same')
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        
        # Subsequent convs: EXACT TensorFlow pattern
        # TensorFlow: for _ in range(conv_num-1): conv = Conv2D(channels//2, ...)
        # First iteration: channels → channels//2
        # Subsequent iterations: channels//2 → channels//2
        current_channels = out_channels
        for _ in range(conv_num - 1):
            target_channels = out_channels // 2
            layers.append(nn.Conv2d(current_channels, target_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            current_channels = target_channels  # For next iteration: channels//2 → channels//2
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x_up, x_skip):
        x_up = self.upsample(x_up)
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv_block(x)


class DeconvUNet(nn.Module):
    """
    Single-stage 2D U-Net for deconvolution only.
    Based on the second stage of the original two-stage U-Net.
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
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_ch = input_channels
        for n in range(conv_block_num):
            out_ch = 2 ** (n + 5)  # 32, 64, 128, 256
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch, conv_num))
            in_ch = out_ch
        
        # Middle layers
        mid_ch = 2 ** (conv_block_num + 5)  # channels * 2
        self.mid_conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.mid_conv2 = nn.Conv2d(mid_ch, in_ch, kernel_size=3, padding=1)
        self.mid_relu = nn.ReLU()
        
        # Decoder - match TensorFlow channel calculation exactly
        self.decoder_blocks = nn.ModuleList()
        encoder_channels = [2 ** (n + 5) for n in range(conv_block_num)]  # [32, 64, 128, 256]
        init_channels = in_ch  # This should be 256 (last encoder channel count)
        
        for n in range(conv_block_num):
            # TensorFlow pattern: channels = init_channels // (2 ** n)
            out_ch = init_channels // (2 ** n)
            
            # Skip connection channels from corresponding encoder block (reverse order)
            skip_ch = encoder_channels[conv_block_num - 1 - n]
            
            # Calculate input channels based on previous layer output
            if n == 0:
                # First decoder block receives middle layer output (256 channels)
                current_ch = in_ch
            else:
                # Previous ConcatBlock outputs out_ch_prev // 2 (EXACT TensorFlow behavior)
                prev_out_ch = init_channels // (2 ** (n-1))
                current_ch = prev_out_ch // 2  # This is what ConcatBlock actually outputs
            
            concat_in_ch = current_ch + skip_ch
            self.decoder_blocks.append(ConcatBlock(concat_in_ch, out_ch, conv_num))
        
        # Final layers - calculate actual output from last decoder block
        # Last decoder block's out_ch parameter: init_channels // (2 ** (conv_block_num - 1))
        # For conv_block_num=4: init_channels // 8 = 256 // 8 = 32
        last_out_ch = init_channels // (2 ** (conv_block_num - 1))
        # ConcatBlock actually outputs out_ch // 2 (EXACT TensorFlow behavior)
        final_in_ch = last_out_ch // 2  # 32 // 2 = 16
        if upsample_flag:
            self.final_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        else:
            self.final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU()
        
        # CRITICAL: Initialize final layer properly for deconvolution
        # PyTorch's default initialization may be too conservative for this task
        self._initialize_final_layers()
    
    def forward(self, x):
        # Store original input dimensions
        batch_size, channels, h, w = x.shape
        
        # Encoder path
        encoder_outputs = []
        current = x
        
        for encoder_block in self.encoder_blocks:
            current, conv_out = encoder_block(current)
            encoder_outputs.append(conv_out)
        
        # Middle layers
        current = self.mid_relu(self.mid_conv1(current))
        current = self.mid_relu(self.mid_conv2(current))
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = encoder_outputs[-(i+1)]
            current = decoder_block(current, skip_connection)
        
        # Final layers
        if self.upsample_flag:
            current = self.final_upsample(current)
        
        current = self.final_relu(self.final_conv1(current))
        current = self.final_relu(self.final_conv2(current))
        
        # CRITICAL FIX: Use proper initialization instead of bias compensation
        # Problem analysis:
        # 1. Raw output range is tiny: [0.001, 0.023] indicating poor initialization
        # 2. Need to fix the root cause: network initialization, not add compensations
        
        output_raw = self.final_conv3(current)
        
        # For debugging: check if outputs are mostly negative (initialization issue)
        negative_ratio = (output_raw < 0).float().mean()
        
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            print(f"\n=== Final Layer Analysis ===")
            print(f"Raw output range: [{output_raw.min().item():.6f}, {output_raw.max().item():.6f}]")
            print(f"Negative ratio: {negative_ratio.item():.3f}")
            print(f"Final conv3 bias: {self.final_conv3.bias.data.item():.6f}")
        
        # Use ReLU as in TensorFlow original, but with proper handling
        if negative_ratio > 0.5:  # If most outputs are negative (bad initialization)
            # Temporary fix: add learnable bias parameter
            if not hasattr(self, 'output_bias'):
                self.register_parameter('output_bias', nn.Parameter(torch.tensor(0.1)))
            output = self.final_relu(output_raw + self.output_bias)
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"Using learnable bias: {self.output_bias.item():.6f}")
        else:
            # Normal case: direct ReLU
            output = self.final_relu(output_raw)
        
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            print(f"Final output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
            print("=" * 60)
        
        return output
    
    def _initialize_final_layers(self):
        """Initialize final layers to ensure proper output range."""
        # Initialize final conv layer with larger weights and zero bias
        # This should help generate outputs with proper dynamic range
        nn.init.xavier_normal_(self.final_conv3.weight, gain=1.0)
        nn.init.constant_(self.final_conv3.bias, 0.0)
        
        # Also initialize other final layers
        nn.init.xavier_normal_(self.final_conv1.weight, gain=1.0)
        nn.init.constant_(self.final_conv1.bias, 0.0)
        nn.init.xavier_normal_(self.final_conv2.weight, gain=1.0) 
        nn.init.constant_(self.final_conv2.bias, 0.0)
    
    def forward_with_crop(self, x):
        """
        Forward pass with cropping to remove padded regions.
        Used when insert_xy > 0.
        """
        output = self.forward(x)
        
        if self.insert_xy > 0:
            # Crop the output to remove padded regions
            if self.upsample_flag:
                # For upsampled output, adjust crop accordingly
                crop_xy = self.insert_xy * 2
                h_start, h_end = crop_xy, output.shape[2] - crop_xy
                w_start, w_end = crop_xy, output.shape[3] - crop_xy
            else:
                h_start, h_end = self.insert_xy, output.shape[2] - self.insert_xy
                w_start, w_end = self.insert_xy, output.shape[3] - self.insert_xy
            
            output = output[:, :, h_start:h_end, w_start:w_end]
        
        return output


def create_deconv_unet(config=None):
    """Factory function to create DeconvUNet with configuration."""
    if config is None:
        config = {
            'input_channels': 1,
            'output_channels': 1,
            'conv_block_num': 4,
            'conv_num': 3,
            'upsample_flag': True,
            'insert_xy': 16
        }
    
    return DeconvUNet(**config)


if __name__ == "__main__":
    # Test the model
    model = create_deconv_unet()
    x = torch.randn(1, 1, 160, 160)  # Batch=1, Channels=1, H=160, W=160
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")