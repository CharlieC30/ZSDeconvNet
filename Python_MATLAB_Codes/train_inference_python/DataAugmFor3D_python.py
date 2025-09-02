#!/usr/bin/env python3
"""
Python implementation of MATLAB DataAugmFor3D.m
Exact replication of MATLAB functionality
"""

import numpy as np
import tifffile
from scipy import ndimage
from pathlib import Path
import glob
import json
import datetime
from scipy.ndimage import rotate

class DataAugmFor3D:
    def __init__(self, 
                 data_path,
                 save_path='data/augmented_data/',
                 seg_x=64,
                 seg_y=64, 
                 seg_z=13,
                 seg_num=10000,
                 rot_flag=True):
        """
        Args:
            data_path: Input data file path
            save_path: Output directory path
            seg_x, seg_y, seg_z: Patch dimensions
            seg_num: Target number of patches
            rot_flag: Enable rotation augmentation
        """
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.seg_x = seg_x
        self.seg_y = seg_y
        self.seg_z = seg_z
        self.seg_num = seg_num
        self.rot_flag = rot_flag
        
        # Threshold parameters (match MATLAB)
        self.thresh_mask = 1e-2
        self.active_range_thresh = 0.5
        self.sum_thresh = 0.01 * seg_x * seg_y * seg_z
        
        # Create output directory with source info and timestamp
        source_name = self.data_path.stem
        now = datetime.datetime.now()
        timestamp = now.strftime('%m%d_%H%M')
        folder_name = f'{source_name}_{timestamp}'
        
        self.save_training_path = self.save_path / folder_name
        self.input_path = self.save_training_path / 'input'
        self.gt_path = self.save_training_path / 'gt'
        
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)
        
        # Store source file information
        self.source_info = {
            'source_file': self.data_path.name,
            'source_path': str(self.data_path),
            'patch_size': [seg_z, seg_x, seg_y],
            'total_samples': seg_num,
            'augmentation': {
                'rotation': rot_flag,
                'flip_ud': True,
                'flip_lr': True,
                'flip_both': True
            },
            'z_sampling': 'interleaved'
        }
        
        # Calculate padding for rotation
        if rot_flag:
            self.halfx = int(seg_x * 1.5 / 2)
            self.halfy = int(seg_y * 1.5 / 2)
            self.tx = self.halfx - seg_x // 2
            self.ty = self.halfy - seg_y // 2
        else:
            self.halfx = seg_x // 2
            self.halfy = seg_y // 2
    
    def xx_norm(self, data):
        """Normalize data to [0,1] range"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-6:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def xx_cal_mask(self, data, sigma=10, threshold=1e-2):
        """DoG mask calculation (exact MATLAB XxCalMask implementation)"""
        # Small Gaussian (foreground)
        fd = ndimage.gaussian_filter(data, sigma=sigma, mode='nearest')
        # Large Gaussian (background)
        bg = ndimage.gaussian_filter(data, sigma=50, mode='nearest')
        # Difference of Gaussians
        mask = fd - bg
        return mask >= threshold
    
    def process_file(self, file_path, n_per_stack):
        """Process single file"""
        # Load data
        data = tifffile.imread(file_path).astype(np.float32)
        
        # Handle 2D data
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        
        # Basic processing
        data[data < 0] = 0
        data = self.xx_norm(data)
        
        # Calculate mask
        cur_thresh = self.thresh_mask
        mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)
        
        # Adjust threshold if not enough valid points
        ntry = 0
        while np.sum(mask) < n_per_stack and ntry < 1000:
            cur_thresh *= 0.8
            mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)
            ntry += 1
            if ntry > 1000:
                break
        
        # Get valid point coordinates (CORRECT ORDER: Y,X,Z to match MATLAB)
        # MATLAB meshgrid: [X,Y,Z] = meshgrid(1:size(data,2), 1:size(data,1), 1:size(data,3))
        # MATLAB point_list(:,1) = Y(mask(:)), point_list(:,2) = X(mask(:)), point_list(:,3) = Z(mask(:))
        z_coords, y_coords, x_coords = np.where(mask)  # np.where returns (Z,Y,X)
        point_list = np.column_stack([y_coords, x_coords, z_coords])  # Reorder to (Y,X,Z) like MATLAB
        
        n_points = len(point_list)
        if n_points == 0:
            return 0
        
        n_total = 0
        n_left = n_per_stack
        max_attempts = 100000
        attempts = 0
        
        while n_left >= 1 and attempts < max_attempts:
            attempts += 1
            
            # Random point selection
            p = np.random.randint(0, n_points)
            y, x, z = point_list[p]  # MATLAB: point_list(p,1), point_list(p,2), point_list(p,3)
            
            # Calculate crop boundaries (exact MATLAB logic)
            # MATLAB uses 1-based indexing, we convert to 0-based
            # Note: MATLAB point_list(p,1) corresponds to Y coordinate (row)
            # Note: MATLAB point_list(p,2) corresponds to X coordinate (column)
            x1_matlab = y - self.halfx + 1  # MATLAB: point_list(p,1) - halfx + 1
            x2_matlab = y + self.halfx      # MATLAB: point_list(p,1) + halfx  
            y1_matlab = x - self.halfy + 1  # MATLAB: point_list(p,2) - halfy + 1
            y2_matlab = x + self.halfy      # MATLAB: point_list(p,2) + halfy
            z1_matlab = z - self.seg_z + 1  # MATLAB: point_list(p,3) - SegZ + 1
            z2_matlab = z + self.seg_z      # MATLAB: point_list(p,3) + SegZ
            
            # Boundary check (MATLAB 1-based)
            if (x1_matlab < 1 or y1_matlab < 1 or z1_matlab < 1 or
                x2_matlab > data.shape[1] or y2_matlab > data.shape[2] or z2_matlab > data.shape[0]):
                continue
            
            # Convert to 0-based indexing
            x1 = x1_matlab - 1; x2 = x2_matlab - 1
            y1 = y1_matlab - 1; y2 = y2_matlab - 1  
            z1 = z1_matlab - 1; z2 = z2_matlab - 1
            
            # Z-axis interleaved sampling (exact MATLAB)
            # MATLAB: input_crop = data(x1:x2, y1:y2, z1+1:2:z2)
            # MATLAB: gt_crop = data(x1:x2, y1:y2, z1:2:z2)
            input_z_indices = np.arange(z1+1, z2+1, 2)  # z1+1:2:z2
            gt_z_indices = np.arange(z1, z2+1, 2)        # z1:2:z2
            
            # Check Z slice count
            if len(input_z_indices) != self.seg_z or len(gt_z_indices) != self.seg_z:
                continue
                
            # Check Z index bounds
            if (np.max(input_z_indices) >= data.shape[0] or 
                np.max(gt_z_indices) >= data.shape[0] or
                np.min(input_z_indices) < 0 or np.min(gt_z_indices) < 0):
                continue
            
            # Extract patches: MATLAB data(x1:x2, y1:y2, z_indices) -> Python data[z_indices, x1:x2+1, y1:y2+1]
            # Note: MATLAB x corresponds to our Y dimension, MATLAB y corresponds to our X dimension
            input_crop = data[input_z_indices, x1:x2+1, y1:y2+1]
            gt_crop = data[gt_z_indices, x1:x2+1, y1:y2+1]
            
            # Transpose to match MATLAB dimension order: (Z,Y,X) -> (Y,X,Z)
            input_crop = np.transpose(input_crop, (1, 2, 0))
            gt_crop = np.transpose(gt_crop, (1, 2, 0))
            
            # Size check and adjustment
            if (input_crop.shape[0] < self.seg_y or input_crop.shape[1] < self.seg_x or
                gt_crop.shape[0] < self.seg_y or gt_crop.shape[1] < self.seg_x):
                continue
            
            input_crop = input_crop[:self.seg_y, :self.seg_x, :self.seg_z]
            gt_crop = gt_crop[:self.seg_y, :self.seg_x, :self.seg_z]
            
            # Quality checks (exact MATLAB)
            p99 = np.percentile(input_crop, 99.9)
            p01 = np.percentile(input_crop, 0.1) + 1e-2
            if p01 > 0:
                active_range = p99 / p01
                if active_range < self.active_range_thresh:
                    continue
            
            sum_value = np.sum(input_crop)
            if sum_value < self.sum_thresh:
                continue
            
            # Generate 4 augmented versions (exact MATLAB)
            augmented_pairs = self.generate_augmentations(input_crop, gt_crop)
            
            # Save all augmented pairs
            for inp, gt in augmented_pairs:
                n_total += 1
                
                # Convert to uint16 (exact MATLAB)
                inp_uint16 = (np.clip(inp, 0, 1) * 65535).astype(np.uint16)
                gt_uint16 = (np.clip(gt, 0, 1) * 65535).astype(np.uint16)
                
                # Convert back to (Z,Y,X) for saving
                inp_save = np.transpose(inp_uint16, (2, 0, 1))
                gt_save = np.transpose(gt_uint16, (2, 0, 1))
                
                # Save files (exact MATLAB naming)
                tifffile.imwrite(str(self.input_path / f'{n_total}.tif'), inp_save)
                tifffile.imwrite(str(self.gt_path / f'{n_total}.tif'), gt_save)
            
            n_left -= 1
        
        return n_total
    
    def generate_augmentations(self, input_crop, gt_crop):
        """Generate 4 augmented versions (exact MATLAB)"""
        augmented_pairs = []
        
        # 1. Original (with optional rotation)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_rot = self.rotate_volume(input_crop, angle)
            gt_rot = self.rotate_volume(gt_crop, angle)
            augmented_pairs.append((inp_rot, gt_rot))
        else:
            augmented_pairs.append((input_crop.copy(), gt_crop.copy()))
        
        # 2. Vertical flip
        inp_flip_ud = np.flip(input_crop, axis=0)
        gt_flip_ud = np.flip(gt_crop, axis=0)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_ud = self.rotate_volume(inp_flip_ud, angle)
            gt_flip_ud = self.rotate_volume(gt_flip_ud, angle)
        augmented_pairs.append((inp_flip_ud, gt_flip_ud))
        
        # 3. Horizontal flip
        inp_flip_lr = np.flip(input_crop, axis=1)
        gt_flip_lr = np.flip(gt_crop, axis=1)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_lr = self.rotate_volume(inp_flip_lr, angle)
            gt_flip_lr = self.rotate_volume(gt_flip_lr, angle)
        augmented_pairs.append((inp_flip_lr, gt_flip_lr))
        
        # 4. Both vertical and horizontal flip
        inp_flip_both = np.flip(np.flip(input_crop, axis=0), axis=1)
        gt_flip_both = np.flip(np.flip(gt_crop, axis=0), axis=1)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_both = self.rotate_volume(inp_flip_both, angle)
            gt_flip_both = self.rotate_volume(gt_flip_both, angle)
        augmented_pairs.append((inp_flip_both, gt_flip_both))
        
        return augmented_pairs
    
    def rotate_volume(self, volume, angle):
        """3D volume rotation (slice-by-slice, matches MATLAB imrotate)"""
        if not self.rot_flag:
            return volume
        
        rotated = np.zeros((self.seg_y, self.seg_x, self.seg_z))
        canvas_size = max(int(self.seg_x * 1.5), int(self.seg_y * 1.5))
        
        for z in range(volume.shape[2]):
            # Place on larger canvas
            canvas = np.zeros((canvas_size, canvas_size))
            start_y = (canvas_size - volume.shape[0]) // 2
            start_x = (canvas_size - volume.shape[1]) // 2
            canvas[start_y:start_y+volume.shape[0], start_x:start_x+volume.shape[1]] = volume[:, :, z]
            
            # Rotate
            slice_rot = rotate(canvas, angle, reshape=False, order=1, mode='constant', cval=0)
            
            # Crop to target size
            center_y, center_x = canvas_size // 2, canvas_size // 2
            crop_y1 = center_y - self.seg_y // 2
            crop_x1 = center_x - self.seg_x // 2
            rotated[:, :, z] = slice_rot[crop_y1:crop_y1+self.seg_y, crop_x1:crop_x1+self.seg_x]
        
        return rotated
    
    def _save_info_file(self, total_generated):
        """Save dataset generation info to JSON file"""
        now = datetime.datetime.now()
        info = {
            **self.source_info,
            'actual_samples_generated': total_generated,
            'generation_date': now.strftime('%Y-%m-%d'),
            'generation_time': now.strftime('%H:%M:%S'),
            'parameters': {
                'thresh_mask': self.thresh_mask,
                'active_range_thresh': self.active_range_thresh,
                'sum_thresh': self.sum_thresh
            }
        }
        
        info_file = self.save_training_path / 'info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def run(self):
        """Execute data augmentation"""
        print(f"Processing {self.data_path.name}")
        
        # Get files
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob('*.tif')) + list(self.data_path.glob('*.tiff'))
        
        if not files:
            print("Error: No TIFF files found")
            return
        
        # Calculate n_per_stack globally (match MATLAB exactly)
        files_count = len(files)
        n_per_stack = max(int(np.ceil(self.seg_num / files_count)), 1)
        
        total_generated = 0
        for file in files:
            n = self.process_file(file, n_per_stack)
            total_generated += n
        
        print(f"Generated {total_generated} training pairs")
        self._save_info_file(total_generated)
        return str(self.save_training_path)


# Example usage
if __name__ == "__main__":
    augmentor = DataAugmFor3D(
        data_path='data/ori_input/iUExM/roiC_crop256_1256.tif',
        save_path='data/augmented_data/iUExM',
        seg_x=64,
        seg_y=64,
        seg_z=13,
        seg_num=10000,
        rot_flag=True
    )
    
    augmentor.run()