"""
Two-stage RCAN3D model for ZS-DeconvNet
PyTorch implementation equivalent to original TensorFlow twostage_RCAN3D.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .blocks import RCAB3D, ResidualGroup3D


class RCAN3D(nn.Module):
    """
    3D Residual Channel Attention Network
    Equivalent to original TensorFlow RCAN3D implementation
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int, int],  # (H, W, D, C)
                 n_resgroups: int = 2,
                 n_rcab: int = 2,
                 n_channels: int = 64,
                 reduction: int = 16,
                 upsample_flag: int = 0,
                 insert_xy: int = 8,
                 insert_z: int = 2):
        """
        Args:
            input_shape: Input shape (H, W, D, C) 
            n_resgroups: Number of residual groups
            n_rcab: Number of RCAB blocks per group
            n_channels: Number of channels
            reduction: Channel reduction for attention
            upsample_flag: 0 or 1, whether to upsample
            insert_xy: XY padding size
            insert_z: Z padding size
        """
        super().__init__()
        
        self.upsample_flag = bool(upsample_flag)
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        
        # Stage 1: Denoising
        self.denoise_input = nn.Conv3d(1, n_channels, 3, padding=1)
        
        denoise_res_groups = []
        for _ in range(n_resgroups):
            denoise_res_groups.append(ResidualGroup3D(n_channels, n_rcab, reduction))
        self.denoise_res_groups = nn.Sequential(*denoise_res_groups)
        
        self.denoise_output = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(n_channels, 1, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Stage 2: Deconvolution  
        self.deconv_input = nn.Conv3d(1, n_channels, 3, padding=1)
        
        deconv_res_groups = []
        for _ in range(n_resgroups):
            deconv_res_groups.append(ResidualGroup3D(n_channels, n_rcab, reduction))
        self.deconv_res_groups = nn.Sequential(*deconv_res_groups)
        
        # Upsampling layer if needed
        if self.upsample_flag:
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
        self.deconv_output = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(n_channels, 1, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through two-stage RCAN3D
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Tuple of (denoised_output, deconvolved_output)
        """
        # Stage 1: Denoising
        conv1 = self.denoise_input(x)
        res1 = conv1
        conv1 = self.denoise_res_groups(conv1)
        conv1 = res1 + conv1  # Residual connection
        denoise_out = self.denoise_output(conv1)
        
        # Stage 2: Deconvolution
        conv2 = self.deconv_input(denoise_out)
        res2 = conv2
        conv2 = self.deconv_res_groups(conv2)
        conv2 = res2 + conv2  # Residual connection
        
        if self.upsample_flag:
            conv2 = self.upsample(conv2)
        
        deconv_out = self.deconv_output(conv2)
        
        return denoise_out, deconv_out
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Factory function matching original interface
def RCAN3D_factory(input_shape: Tuple[int, int, int, int],
                   upsample_flag: int = 0,
                   insert_xy: int = 8,
                   insert_z: int = 2,
                   n_ResGroup: int = 2,
                   n_RCAB: int = 2) -> RCAN3D:
    """
    Factory function matching original TensorFlow interface
    
    Args:
        input_shape: Input shape (H, W, D, C)
        upsample_flag: Whether to upsample (0 or 1)
        insert_xy: XY padding
        insert_z: Z padding  
        n_ResGroup: Number of residual groups
        n_RCAB: Number of RCAB blocks per group
    
    Returns:
        RCAN3D model instance
    """
    return RCAN3D(
        input_shape=input_shape,
        n_resgroups=n_ResGroup,
        n_rcab=n_RCAB,
        upsample_flag=upsample_flag,
        insert_xy=insert_xy,
        insert_z=insert_z
    )


__all__ = ['RCAN3D', 'RCAN3D_factory']