"""
Simplified data loading utilities for ZS-DeconvNet PyTorch implementation
Matches original TensorFlow DataLoader function approach
"""

import numpy as np
import torch
import glob
import os
from pathlib import Path
from typing import Tuple, List
import tifffile
from .utils import prctile_norm


def normalize_image(img: np.ndarray, norm_flag: int) -> np.ndarray:
    """Normalize image based on norm_flag (matching original)"""
    if norm_flag == 1:
        return prctile_norm(img)
    elif norm_flag == 0:
        return img / 65535.0
    elif norm_flag == 2:
        max_val = np.max(img)
        return img / max_val if max_val > 0 else img
    else:
        return img


def read_tiff_stack(file_path: str) -> np.ndarray:
    """Read TIFF stack (matching original mimread)"""
    return np.array(tifffile.imread(file_path)).astype(np.float32)


def apply_padding(input_batch: np.ndarray, insert_xy: int, insert_z: int) -> np.ndarray:
    """Apply padding matching original TensorFlow concatenation logic"""
    batch_size, h, w, d, c = input_batch.shape
    
    # Z-direction padding
    insert_shape = np.zeros([batch_size, h, w, insert_z, c])
    input_batch = np.concatenate((insert_shape, input_batch, insert_shape), axis=3)
    
    # Y-direction padding
    _, h, w, d, c = input_batch.shape
    insert_shape = np.zeros([batch_size, insert_xy, w, d, c])
    input_batch = np.concatenate((insert_shape, input_batch, insert_shape), axis=1)
    
    # X-direction padding
    _, h, w, d, c = input_batch.shape
    insert_shape = np.zeros([batch_size, h, insert_xy, d, c])
    input_batch = np.concatenate((insert_shape, input_batch, insert_shape), axis=2)
    
    return input_batch


class SimplifiedDataModule:
    """Simplified data module matching original TensorFlow approach"""
    
    def __init__(self, 
                 data_dir: str,
                 folder: str = "",
                 input_shape: Tuple[int, int, int] = (64, 64, 13),
                 insert_xy: int = 8,
                 insert_z: int = 2,
                 batch_size: int = 3,
                 norm_flag: int = 0,
                 load_all_data: bool = True):
        
        self.data_dir = data_dir
        self.folder = folder
        self.input_shape = input_shape  # (input_y, input_x, input_z)
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        self.batch_size = batch_size
        self.norm_flag = norm_flag
        self.load_all_data = load_all_data
        
        # Set up paths (matching original)
        if folder:
            self.train_images_path = os.path.join(data_dir, folder, 'input')
            self.train_gt_path = os.path.join(data_dir, folder, 'gt')
        else:
            self.train_images_path = os.path.join(data_dir, 'input')
            self.train_gt_path = os.path.join(data_dir, 'gt')
        
        # Find all image files
        self.images_path = glob.glob(os.path.join(self.train_images_path, '*'))
        print(f"Found {len(self.images_path)} training images in {self.train_images_path}")
        
        if len(self.images_path) == 0:
            raise ValueError(f"No images found in {self.train_images_path}")
        
        # Pre-load all data if requested (matching original load_all_data logic)
        if load_all_data:
            print("Loading all training data into memory...")
            self.inputs, self.gts = self._load_all_data()
            self.num_total = len(self.inputs)
            print(f"Loaded {self.num_total} training samples")
    
    def _load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all training data into memory"""
        input_y, input_x, input_z = self.input_shape
        all_inputs = []
        all_gts = []
        
        for img_path in self.images_path:
            gt_path = img_path.replace(self.train_images_path, self.train_gt_path)
            
            if os.path.exists(gt_path):
                img = read_tiff_stack(img_path)
                gt = read_tiff_stack(gt_path)
                
                img = normalize_image(img, self.norm_flag)
                gt = normalize_image(gt, self.norm_flag)
                
                # Reshape and transpose to match original (matching original order='F')
                img = np.reshape(img, (1, input_z, input_y, input_x), order='F')
                gt = np.reshape(gt, (1, input_z, input_y, input_x), order='F')
                
                # Transpose to match original (0, 2, 3, 1) -> (B, H, W, D)
                img = np.transpose(img, (0, 2, 3, 1))
                gt = np.transpose(gt, (0, 2, 3, 1))
                
                all_inputs.append(img[0])
                all_gts.append(gt[0])
        
        return np.array(all_inputs, dtype=np.float32), np.array(all_gts, dtype=np.float32)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training batch (matching original training loop)"""
        if self.load_all_data:
            # Random sample from pre-loaded data (matching original index = np.random.choice(num_total,batch_size))
            indices = np.random.choice(self.num_total, self.batch_size, replace=False)
            input_batch = self.inputs[indices]
            gt_batch = self.gts[indices]
        else:
            # Use simple data loader for on-demand loading
            input_batch, gt_batch = self._load_batch_on_demand()
        
        # Add channel dimension
        input_batch = np.expand_dims(input_batch, axis=-1)  # (B, H, W, D, 1)
        gt_batch = np.expand_dims(gt_batch, axis=-1)
        
        # Apply padding (matching original concatenation logic)
        input_batch = apply_padding(input_batch, self.insert_xy, self.insert_z)
        
        # Convert to PyTorch tensors with proper dimension order for PyTorch (B, C, D, H, W)
        input_tensor = torch.from_numpy(input_batch).permute(0, 4, 3, 1, 2).float()
        gt_tensor = torch.from_numpy(gt_batch).permute(0, 4, 3, 1, 2).float()
        
        return input_tensor, gt_tensor
    
    def _load_batch_on_demand(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load batch on demand (matching original DataLoader function)"""
        input_y, input_x, input_z = self.input_shape
        
        # Random sample batch_size images (matching original)
        batch_paths = np.random.choice(self.images_path, size=self.batch_size, replace=False)
        
        input_batch = []
        gt_batch = []
        
        for path in batch_paths:
            gt_path_full = path.replace(self.train_images_path, self.train_gt_path)
            
            # Ensure GT file exists (matching original while loop)
            while not os.path.exists(gt_path_full):
                path = np.random.choice(self.images_path, size=1)[0]
                gt_path_full = path.replace(self.train_images_path, self.train_gt_path)
            
            # Load images (matching original mimread)
            img = read_tiff_stack(path)
            gt = read_tiff_stack(gt_path_full)
            
            # Normalize (matching original logic)
            img = normalize_image(img, self.norm_flag)
            gt = normalize_image(gt, self.norm_flag)
            
            input_batch.append(img)
            gt_batch.append(gt)
        
        # Convert to numpy arrays
        input_batch = np.array(input_batch, dtype=np.float32)
        gt_batch = np.array(gt_batch, dtype=np.float32)
        
        # Reshape to match original order='F' logic
        input_batch = np.reshape(input_batch, (self.batch_size, 1, input_z, input_y, input_x), order='F')
        gt_batch = np.reshape(gt_batch, (self.batch_size, 1, input_z, input_y, input_x), order='F')
        
        # Transpose to match original (0, 3, 4, 2, 1) -> (B, H, W, D, C)
        input_batch = np.transpose(input_batch, (0, 3, 4, 2, 1))
        gt_batch = np.transpose(gt_batch, (0, 3, 4, 2, 1))
        
        # Remove the channel dimension for now (will be added back in get_batch)
        return input_batch[..., 0], gt_batch[..., 0]


# Legacy function for compatibility with original code style
def DataLoader(images_path: List[str], 
               data_path: str, 
               gt_path: str,
               batch_size: int, 
               norm_flag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy DataLoader function matching original interface
    Used for backward compatibility
    
    Returns:
        Tuple of (batch_images, batch_gts)
    """
    # Random sample batch_size images
    batch_paths = np.random.choice(images_path, size=batch_size, replace=False)
    
    images = []
    gts = []
    
    for path in batch_paths:
        gt_path_full = path.replace(data_path, gt_path)
        
        # Ensure GT file exists
        while not os.path.exists(gt_path_full):
            path = np.random.choice(images_path, size=1)[0]
            gt_path_full = path.replace(data_path, gt_path)
        
        # Load images
        img = read_tiff_stack(path)
        gt = read_tiff_stack(gt_path_full)
        
        # Normalize
        img = normalize_image(img, norm_flag)
        gt = normalize_image(gt, norm_flag)
        
        images.append(img)
        gts.append(gt)
    
    return np.array(images, dtype=np.float32), np.array(gts, dtype=np.float32)


__all__ = [
    'SimplifiedDataModule', 
    'DataLoader',  # Legacy
    'normalize_image',
    'read_tiff_stack',
    'apply_padding'
]