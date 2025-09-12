"""
Two-stage UNet3D model for ZS-DeconvNet
PyTorch implementation equivalent to original TensorFlow twostage_Unet3D.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .blocks import DoubleConv3D, Down3D, Up3D, OutConv3D


class Unet3D(nn.Module):
    """
    3D U-Net architecture for ZS-DeconvNet
    Two-stage implementation: denoising + deconvolution
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int, int],  # (H, W, D, C)
                 n_channels: int = 1,
                 n_classes: int = 1,
                 features: list = [64, 128, 256, 512],
                 upsample_flag: int = 0,
                 insert_xy: int = 8,
                 insert_z: int = 2,
                 trilinear: bool = True):
        """
        Args:
            input_shape: Input shape (H, W, D, C)
            n_channels: Number of input channels
            n_classes: Number of output classes
            features: Feature map sizes for each level
            upsample_flag: 0 or 1, whether to upsample
            insert_xy: XY padding size
            insert_z: Z padding size
            trilinear: Use trilinear upsampling
        """
        super().__init__()
        
        self.upsample_flag = bool(upsample_flag)
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        self.trilinear = trilinear
        
        # Stage 1: Denoising U-Net
        self.denoise_unet = self._build_unet_stage(n_channels, n_classes, features, trilinear)
        
        # Stage 2: Deconvolution U-Net
        self.deconv_unet = self._build_unet_stage(n_channels, n_classes, features, trilinear)
        
        # Upsampling layer if needed
        if self.upsample_flag:
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            
        self._initialize_weights()
    
    def _build_unet_stage(self, n_channels: int, n_classes: int, features: list, trilinear: bool):
        """Build a single U-Net stage"""
        
        # Encoder (Down path)
        inc = DoubleConv3D(n_channels, features[0])
        
        downs = nn.ModuleList()
        for i in range(len(features) - 1):
            downs.append(Down3D(features[i], features[i + 1]))
        
        # Decoder (Up path)  
        ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            ups.append(Up3D(features[i], features[i - 1], trilinear))
        
        # Output convolution
        outc = OutConv3D(features[0], n_classes)
        
        return nn.ModuleDict({
            'inc': inc,
            'downs': downs,
            'ups': ups,
            'outc': outc
        })
    
    def _forward_unet_stage(self, x: torch.Tensor, stage: nn.ModuleDict) -> torch.Tensor:
        """Forward pass through a single U-Net stage"""
        # Input convolution
        x1 = stage['inc'](x)
        
        # Encoder path
        encoder_features = [x1]
        for down in stage['downs']:
            x1 = down(x1)
            encoder_features.append(x1)
        
        # Decoder path
        x = encoder_features[-1]  # Bottom features
        for i, up in enumerate(stage['ups']):
            skip_connection = encoder_features[-(i + 2)]  # Get corresponding encoder feature
            x = up(x, skip_connection)
        
        # Output
        return stage['outc'](x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through two-stage UNet3D
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Tuple of (denoised_output, deconvolved_output)
        """
        # Stage 1: Denoising
        denoise_out = self._forward_unet_stage(x, self.denoise_unet)
        
        # Stage 2: Deconvolution
        deconv_out = self._forward_unet_stage(denoise_out, self.deconv_unet)
        
        # Apply upsampling if needed
        if self.upsample_flag:
            deconv_out = self.upsample(deconv_out)
        
        return denoise_out, deconv_out
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SimpleUnet3D(nn.Module):
    """
    Simplified 3D U-Net for faster training/inference
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int, int],
                 upsample_flag: int = 0,
                 insert_xy: int = 8,
                 insert_z: int = 2):
        super().__init__()
        
        self.upsample_flag = bool(upsample_flag)
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        
        # Simple encoder-decoder architecture
        # Stage 1: Denoising
        self.denoise_encoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.denoise_decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 1, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Stage 2: Deconvolution
        self.deconv_encoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.deconv_decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 1, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        if self.upsample_flag:
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Stage 1: Denoising
        encoded1 = self.denoise_encoder(x)
        denoise_out = self.denoise_decoder(encoded1)
        
        # Stage 2: Deconvolution
        encoded2 = self.deconv_encoder(denoise_out)
        deconv_out = self.deconv_decoder(encoded2)
        
        if self.upsample_flag:
            deconv_out = self.upsample(deconv_out)
        
        return denoise_out, deconv_out
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Factory function matching original interface
def Unet(input_shape: Tuple[int, int, int, int],
         upsample_flag: int = 0,
         insert_xy: int = 8,
         insert_z: int = 2,
         simple: bool = False) -> nn.Module:
    """
    Factory function matching original TensorFlow interface
    
    Args:
        input_shape: Input shape (H, W, D, C)
        upsample_flag: Whether to upsample (0 or 1)
        insert_xy: XY padding
        insert_z: Z padding
        simple: Use simplified U-Net architecture
    
    Returns:
        UNet3D model instance
    """
    if simple:
        return SimpleUnet3D(
            input_shape=input_shape,
            upsample_flag=upsample_flag,
            insert_xy=insert_xy,
            insert_z=insert_z
        )
    else:
        return Unet3D(
            input_shape=input_shape,
            upsample_flag=upsample_flag,
            insert_xy=insert_xy,
            insert_z=insert_z
        )


__all__ = ['Unet3D', 'SimpleUnet3D', 'Unet']