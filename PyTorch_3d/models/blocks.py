"""
Common building blocks for 3D neural networks
PyTorch implementation of RCAN and U-Net components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GlobalAveragePooling3D(nn.Module):
    """
    Global Average Pooling for 3D tensors
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Pooled tensor (B, C, 1, 1, 1)
        """
        return F.adaptive_avg_pool3d(x, (1, 1, 1))


class ChannelAttention3D(nn.Module):
    """
    Channel Attention mechanism for 3D tensors
    """
    
    def __init__(self, n_channels: int, reduction: int = 16):
        super().__init__()
        self.n_channels = n_channels
        self.reduction = reduction
        
        self.gap = GlobalAveragePooling3D()
        self.fc1 = nn.Conv3d(n_channels, n_channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(n_channels // reduction, n_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Attention-weighted tensor
        """
        # Global average pooling
        w = self.gap(x)  # (B, C, 1, 1, 1)
        
        # Channel attention
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        
        # Apply attention
        return x * w


class RCAB3D(nn.Module):
    """
    Residual Channel Attention Block for 3D
    """
    
    def __init__(self, n_channels: int = 64, reduction: int = 16, bias: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=3, 
                              padding=1, bias=bias)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=3, 
                              padding=1, bias=bias)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.ca = ChannelAttention3D(n_channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Output tensor with residual connection
        """
        residual = x
        
        out = self.leaky_relu1(self.conv1(x))
        out = self.leaky_relu2(self.conv2(out))
        out = self.ca(out)
        
        return residual + out


class ResidualGroup3D(nn.Module):
    """
    Residual Group containing multiple RCAB blocks
    """
    
    def __init__(self, n_channels: int = 64, n_rcab: int = 5, reduction: int = 16):
        super().__init__()
        
        rcab_layers = []
        for _ in range(n_rcab):
            rcab_layers.append(RCAB3D(n_channels, reduction))
        
        self.rcab_blocks = nn.Sequential(*rcab_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            Output tensor after residual group processing
        """
        return self.rcab_blocks(x)


class ConvBlock3D(nn.Module):
    """
    Basic 3D convolution block with activation
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 activation: str = 'leaky_relu', bias: bool = True):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, 
                             kernel_size=kernel_size, stride=stride, 
                             padding=padding, bias=bias)
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block (commonly used in U-Net)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 mid_channels: Optional[int] = None, bias: bool = True):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Upscaling then double conv
    """
    
    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                        kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size differences
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    Output convolution layer
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


__all__ = [
    'GlobalAveragePooling3D',
    'ChannelAttention3D', 
    'RCAB3D',
    'ResidualGroup3D',
    'ConvBlock3D',
    'DoubleConv3D',
    'Down3D',
    'Up3D',
    'OutConv3D'
]