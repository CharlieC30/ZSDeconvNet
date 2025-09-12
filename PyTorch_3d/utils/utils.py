"""
Independent utility functions for ZS-DeconvNet PyTorch implementation
Standalone implementation without dependencies on original code
"""

import numpy as np
import torch
import tifffile
from pathlib import Path
from typing import Union, Tuple, Optional
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import struct


def prctile_norm(x: np.ndarray, min_prc: float = 0, max_prc: float = 100) -> np.ndarray:
    """
    Percentile normalization
    
    Args:
        x: Input array
        min_prc: Minimum percentile
        max_prc: Maximum percentile
    
    Returns:
        Normalized array in range [0, 1]
    """
    min_val = np.percentile(x, min_prc)
    max_val = np.percentile(x, max_prc)
    
    # Avoid division by zero
    if max_val - min_val < 1e-7:
        return np.zeros_like(x)
    
    y = (x - min_val) / (max_val - min_val)
    y = np.clip(y, 0, 1)
    return y


def normalize_image(img: np.ndarray, norm_flag: int = 0) -> np.ndarray:
    """
    Image normalization
    
    Args:
        img: Input image
        norm_flag: 0=/65535, 1=minmax, 2=/max
    
    Returns:
        Normalized image
    """
    if norm_flag == 1:
        return prctile_norm(img)
    elif norm_flag == 0:
        return img / 65535.0
    elif norm_flag == 2:
        max_val = np.max(img)
        if max_val > 0:
            return img / max_val
    return img


def read_tiff_stack(filepath: Union[str, Path]) -> np.ndarray:
    """
    Read TIFF stack
    
    Args:
        filepath: Path to TIFF file
    
    Returns:
        3D numpy array (Z, Y, X)
    """
    img = tifffile.imread(filepath)
    
    # Ensure 3D array
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    
    return img.astype(np.float32)


def save_tiff_stack(filepath: Union[str, Path], img: np.ndarray, dtype: str = 'uint16'):
    """
    Save TIFF stack
    
    Args:
        filepath: Output file path
        img: 3D numpy array (Z, Y, X)
        dtype: Output data type
    """
    if dtype == 'uint16':
        img = np.clip(img * 65535, 0, 65535).astype(np.uint16)
    elif dtype == 'uint8':
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    tifffile.imwrite(filepath, img)


def crop_center_3d(img: np.ndarray, crop_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop from the center of 3D image
    
    Args:
        img: Input image (Z, Y, X)
        crop_size: Crop size (z, y, x)
    
    Returns:
        Cropped image
    """
    z, y, x = img.shape
    crop_z, crop_y, crop_x = crop_size
    
    start_z = (z - crop_z) // 2
    start_y = (y - crop_y) // 2
    start_x = (x - crop_x) // 2
    
    return img[start_z:start_z + crop_z, 
               start_y:start_y + crop_y,
               start_x:start_x + crop_x]


def pad_3d(img: np.ndarray, padding: Union[int, Tuple[int, int, int]], 
           mode: str = 'constant', constant_value: float = 0) -> np.ndarray:
    """
    3D image padding
    
    Args:
        img: Input image (Z, Y, X)
        padding: Padding size
        mode: Padding mode
        constant_value: Fill value for constant mode
    
    Returns:
        Padded image
    """
    if isinstance(padding, int):
        pad_z = pad_y = pad_x = padding
    else:
        pad_z, pad_y, pad_x = padding
    
    pad_width = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))
    
    if mode == 'constant':
        return np.pad(img, pad_width, mode=mode, constant_values=constant_value)
    else:
        return np.pad(img, pad_width, mode=mode)


def calculate_otf_from_psf(psf: np.ndarray) -> np.ndarray:
    """
    Calculate OTF (Optical Transfer Function) from PSF
    
    Args:
        psf: Point Spread Function
    
    Returns:
        OTF
    """
    # Normalize PSF
    psf = psf / np.sum(psf)
    
    # Calculate OTF using FFT
    otf = np.fft.fftn(psf)
    otf = np.fft.fftshift(otf)
    
    return np.abs(otf)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        numpy array
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        array: numpy array
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array).float()
    if device:
        tensor = tensor.to(device)
    return tensor


__all__ = [
    'prctile_norm',
    'normalize_image',
    'read_tiff_stack',
    'save_tiff_stack',
    'crop_center_3d',
    'pad_3d',
    'calculate_otf_from_psf',
    'tensor_to_numpy',
    'numpy_to_tensor'
]