import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import tifffile
from typing import Optional, List, Tuple
import random
from sklearn.model_selection import train_test_split


class TiffSliceDataset(Dataset):
    """
    Dataset that treats each z-slice of 3D TIFF files as independent 2D images.
    """
    
    def __init__(self, 
                 input_paths: List[str], 
                 target_paths: List[str],
                 patch_size: int = 128,
                 insert_xy: int = 16,
                 normalize: bool = True,
                 augment: bool = False):
        """
        Initialize dataset.
        
        Args:
            input_paths: List of paths to input 3D TIFF files
            target_paths: List of paths to target 3D TIFF files
            patch_size: Size of image patches to extract
            insert_xy: Padding size for the patches
            normalize: Whether to normalize images to [0, 1]
            augment: Whether to apply data augmentation
        """
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.patch_size = patch_size
        self.insert_xy = insert_xy
        self.normalize = normalize
        self.augment = augment
        
        # Build slice index mapping
        self._build_slice_mapping()
        
    def _build_slice_mapping(self):
        """Build mapping from dataset index to (file_index, slice_index)."""
        self.slice_mapping = []
        
        for file_idx, (input_path, target_path) in enumerate(zip(self.input_paths, self.target_paths)):
            # Read file to get number of slices
            input_img = tifffile.imread(input_path)
            target_img = tifffile.imread(target_path)
            
            # Handle both 2D and 3D cases
            if len(input_img.shape) == 2:
                num_slices = 1
            else:
                num_slices = input_img.shape[0]
                
            # Verify target has same number of slices
            if len(target_img.shape) == 3:
                assert target_img.shape[0] == num_slices, f"Mismatch in slices: {input_path} vs {target_path}"
            elif len(target_img.shape) == 2 and num_slices > 1:
                raise ValueError(f"Input is 3D but target is 2D: {target_path}")
            
            # Add slice mappings
            for slice_idx in range(num_slices):
                self.slice_mapping.append((file_idx, slice_idx))
    
    def __len__(self):
        return len(self.slice_mapping)
    
    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_mapping[idx]
        
        # Load images
        input_img = tifffile.imread(self.input_paths[file_idx])
        target_img = tifffile.imread(self.target_paths[file_idx])
        
        # Extract slice
        if len(input_img.shape) == 3:
            input_slice = input_img[slice_idx]
            target_slice = target_img[slice_idx]
        else:
            input_slice = input_img
            target_slice = target_img
        
        # Convert to float32
        input_slice = input_slice.astype(np.float32)
        target_slice = target_slice.astype(np.float32)
        
        # Normalize if requested
        if self.normalize:
            input_slice = self._normalize_image(input_slice)
            target_slice = self._normalize_image(target_slice)
        
        # Extract patch
        input_patch, target_patch = self._extract_patch(input_slice, target_slice)
        
        # Apply augmentation if requested
        if self.augment:
            input_patch, target_patch = self._apply_augmentation(input_patch, target_patch)
        
        # Add padding for network input
        input_padded = self._add_padding(input_patch)
        
        # Convert to torch tensors and add channel dimension
        # Make sure arrays are contiguous to avoid negative stride issues
        input_padded = np.ascontiguousarray(input_padded)
        target_patch = np.ascontiguousarray(target_patch)
        
        input_tensor = torch.from_numpy(input_padded).unsqueeze(0)  # Shape: (1, H+2*insert_xy, W+2*insert_xy)
        target_tensor = torch.from_numpy(target_patch).unsqueeze(0)  # Shape: (1, H, W)
        
        return input_tensor, target_tensor
    
    def _normalize_image(self, img):
        """Normalize image using percentile normalization."""
        min_val = np.percentile(img, 0)
        max_val = np.percentile(img, 100)
        normalized = (img - min_val) / (max_val - min_val + 1e-7)
        normalized = np.clip(normalized, 0, 1)
        return normalized
    
    def _extract_patch(self, input_img, target_img):
        """Extract random patch from images."""
        h, w = input_img.shape
        
        # If image is smaller than patch size, pad it
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            input_img = np.pad(input_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = input_img.shape
        
        # Random crop
        max_h = h - self.patch_size
        max_w = w - self.patch_size
        
        start_h = random.randint(0, max_h) if max_h > 0 else 0
        start_w = random.randint(0, max_w) if max_w > 0 else 0
        
        input_patch = input_img[start_h:start_h + self.patch_size, 
                               start_w:start_w + self.patch_size]
        target_patch = target_img[start_h:start_h + self.patch_size, 
                                 start_w:start_w + self.patch_size]
        
        return input_patch, target_patch
    
    def _add_padding(self, img):
        """Add padding around image for network input."""
        pad_width = ((self.insert_xy, self.insert_xy), (self.insert_xy, self.insert_xy))
        return np.pad(img, pad_width, mode='constant', constant_values=0)
    
    def _apply_augmentation(self, input_img, target_img):
        """Apply data augmentation."""
        # Random rotation (0, 90, 180, 270 degrees)
        rot_mode = random.choice([0, 1, 2, 3])
        if rot_mode > 0:
            input_img = np.rot90(input_img, rot_mode).copy()
            target_img = np.rot90(target_img, rot_mode).copy()
        
        # Random flipping
        if random.random() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()
        
        if random.random() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()
        
        return input_img, target_img


class DeconvDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for deconvolution training.
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 patch_size: int = 128,
                 insert_xy: int = 16,
                 num_workers: int = 4,
                 train_val_split: float = 0.8,
                 seed: int = 42):
        """
        Initialize DataModule.
        
        Args:
            data_dir: Directory containing Train/ subdirectory
            batch_size: Batch size for training
            patch_size: Size of image patches
            insert_xy: Padding size
            num_workers: Number of data loading workers
            train_val_split: Fraction of data to use for training
            seed: Random seed for reproducible splits
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.insert_xy = insert_xy
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        
        # Dataset paths
        self.train_input_dir = os.path.join(data_dir, 'Train')
        self.inference_input_dir = os.path.join(data_dir, 'Inference')
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        
        if stage == 'fit' or stage is None:
            # Find all TIFF files in train directory
            train_files = glob.glob(os.path.join(self.train_input_dir, '*.tif'))
            train_files.extend(glob.glob(os.path.join(self.train_input_dir, '*.tiff')))
            train_files = sorted(train_files)
            
            if len(train_files) == 0:
                raise ValueError(f"No TIFF files found in {self.train_input_dir}")
            
            # For this implementation, we assume the same files are used as both input and target
            # In practice, you might have separate input and ground truth directories
            input_files = train_files
            target_files = train_files  # Modify this based on your data structure
            
            # Split into train and validation
            train_input, val_input, train_target, val_target = train_test_split(
                input_files, target_files, 
                train_size=self.train_val_split, 
                random_state=self.seed
            )
            
            # Create datasets
            self.train_dataset = TiffSliceDataset(
                train_input, train_target,
                patch_size=self.patch_size,
                insert_xy=self.insert_xy,
                augment=True
            )
            
            self.val_dataset = TiffSliceDataset(
                val_input, val_target,
                patch_size=self.patch_size,
                insert_xy=self.insert_xy,
                augment=False
            )
        
        if stage == 'test' or stage == 'predict':
            # Setup inference dataset
            inference_files = glob.glob(os.path.join(self.inference_input_dir, '*.tif'))
            inference_files.extend(glob.glob(os.path.join(self.inference_input_dir, '*.tiff')))
            inference_files = sorted(inference_files)
            
            if len(inference_files) == 0:
                raise ValueError(f"No TIFF files found in {self.inference_input_dir}")
            
            self.test_dataset = TiffSliceDataset(
                inference_files, inference_files,  # Use same files for input and target
                patch_size=self.patch_size,
                insert_xy=self.insert_xy,
                augment=False
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one image at a time for inference
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()


class InferenceDataset(Dataset):
    """
    Dataset for inference on full images (not patches).
    """
    
    def __init__(self, image_paths: List[str], insert_xy: int = 16):
        self.image_paths = image_paths
        self.insert_xy = insert_xy
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = tifffile.imread(img_path).astype(np.float32)
        
        # Normalize
        img = self._normalize_image(img)
        
        # If 3D, we'll process each slice
        if len(img.shape) == 3:
            # Process all slices
            slices = []
            for i in range(img.shape[0]):
                slice_img = img[i]
                slice_padded = self._add_padding(slice_img)
                slice_tensor = torch.from_numpy(slice_padded).unsqueeze(0)
                slices.append(slice_tensor)
            
            # Stack slices
            img_tensor = torch.stack(slices, dim=0)  # Shape: (num_slices, 1, H+2*pad, W+2*pad)
        else:
            # 2D image
            img_padded = self._add_padding(img)
            img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H+2*pad, W+2*pad)
        
        return img_tensor, img_path
    
    def _normalize_image(self, img):
        """Normalize image using percentile normalization."""
        min_val = np.percentile(img, 0)
        max_val = np.percentile(img, 100)
        normalized = (img - min_val) / (max_val - min_val + 1e-7)
        normalized = np.clip(normalized, 0, 1)
        return normalized
    
    def _add_padding(self, img):
        """Add padding around image."""
        pad_width = ((self.insert_xy, self.insert_xy), (self.insert_xy, self.insert_xy))
        return np.pad(img, pad_width, mode='constant', constant_values=0)


if __name__ == "__main__":
    # Test the DataModule
    data_dir = "/home/aero/charliechang/projects/ZS-DeconvNet/PyTorch_Deconv/Data"
    
    dm = DeconvDataModule(
        data_dir=data_dir,
        batch_size=2,
        patch_size=128,
        insert_xy=16
    )
    
    # Setup for training
    dm.setup('fit')
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    print(f"Number of training batches: {len(train_loader)}")
    
    # Get a sample batch
    for batch in train_loader:
        input_batch, target_batch = batch
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Target batch shape: {target_batch.shape}")
        break