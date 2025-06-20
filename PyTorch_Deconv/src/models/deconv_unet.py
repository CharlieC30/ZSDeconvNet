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
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_out = self.conv_block(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        layers = []
        # All conv layers use the same output channels (no reduction)
        for i in range(conv_num):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
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
        self.mid_relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        current_ch = in_ch
        encoder_channels = [2 ** (n + 5) for n in range(conv_block_num)]  # [32, 64, 128, 256]
        
        for n in range(conv_block_num):
            # Skip connection channels from corresponding encoder block (reverse order)
            skip_ch = encoder_channels[conv_block_num - 1 - n]
            # Input channels = upsampled current + skip connection
            concat_in_ch = current_ch + skip_ch
            # Output channels decrease by half each step
            out_ch = current_ch // 2
            self.decoder_blocks.append(ConcatBlock(concat_in_ch, out_ch, conv_num))
            current_ch = out_ch
        
        # Final layers
        final_in_ch = current_ch  # Use the final channel count from decoder
        if upsample_flag:
            self.final_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        else:
            self.final_conv1 = nn.Conv2d(final_in_ch, 128, kernel_size=3, padding=1)
        
        self.final_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.final_conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU(inplace=True)
    
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
        output = self.final_conv3(current)  # No ReLU on final output
        
        return output
    
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


