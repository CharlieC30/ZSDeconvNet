#!/usr/bin/env python3
"""
Python 版本的 DataAugmFor3D.m
完整複製 MATLAB 版本的所有功能
"""

import numpy as np
import tifffile
from scipy import ndimage
from pathlib import Path
import glob
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import rotate

class DataAugmFor3D:
    def __init__(self, 
                 data_path,
                 save_path='../your_augmented_datasets/',
                 seg_x=64,
                 seg_y=64, 
                 seg_z=13,
                 seg_num=10000,
                 rot_flag=True):
        """
        參數與 MATLAB 版本完全對應
        """
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.seg_x = seg_x
        self.seg_y = seg_y
        self.seg_z = seg_z
        self.seg_num = seg_num
        self.rot_flag = rot_flag
        
        # 閾值參數（與 MATLAB 一致）
        self.thresh_mask = 1e-2
        self.active_range_thresh = 0.5
        self.sum_thresh = 0.01 * seg_x * seg_y * seg_z
        
        # 建立輸出目錄
        self.save_training_path = self.save_path / f'Zsize{seg_z}_Xsize{seg_x}'
        self.input_path = self.save_training_path / 'input'
        self.gt_path = self.save_training_path / 'gt'
        
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)
        
        # 計算 padding（用於旋轉）
        if rot_flag:
            self.halfx = int(seg_x * 1.5 / 2)
            self.halfy = int(seg_y * 1.5 / 2)
            self.tx = self.halfx - seg_x // 2
            self.ty = self.halfy - seg_y // 2
        else:
            self.halfx = seg_x // 2
            self.halfy = seg_y // 2
    
    def xx_norm(self, data):
        """複製 XxNorm 功能：正規化到 [0,1]"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-6:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def xx_cal_mask(self, data, sigma=10, threshold=1e-2):
        """
        複製 XxCalMask 功能：計算有效區域遮罩
        
        Args:
            data: 3D 影像
            sigma: 高斯濾波參數
            threshold: 閾值
        """
        # 高斯濾波平滑
        smoothed = ndimage.gaussian_filter(data, sigma=sigma)
        
        # 計算遮罩
        mask = smoothed > threshold
        
        # 形態學操作：去除小區域
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=2)
        mask = binary_erosion(mask, iterations=1)
        
        return mask
    
    def process_file(self, file_path):
        """處理單個檔案"""
        print(f"處理檔案: {file_path}")
        
        # 讀取資料
        data = tifffile.imread(file_path).astype(np.float32)
        
        # 處理維度
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]  # 添加 Z 維度
        
        print(f"資料形狀: {data.shape} (Z, Y, X)")
        
        # 基本處理
        data[data < 0] = 0
        data = self.xx_norm(data)
        
        # 計算遮罩
        cur_thresh = self.thresh_mask
        mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)
        
        # 如果有效區域太少，降低閾值
        ntry = 0
        while np.sum(mask) < 100 and ntry < 1000:  # 至少需要 100 個有效點
            cur_thresh *= 0.8
            mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)
            ntry += 1
        
        # 獲取有效點位置
        valid_points = np.where(mask)
        if len(valid_points[0]) == 0:
            print("警告：沒有找到有效區域")
            return 0
        
        point_list = np.column_stack(valid_points)
        n_points = len(point_list)
        print(f"找到 {n_points} 個有效點")
        
        # 計算每個檔案要生成的 patch 數量
        if self.data_path.is_file():
            n_per_file = self.seg_num  # 單檔案時，生成所有 patches
        else:
            files_count = len(glob.glob(str(self.data_path) + "/*"))
            n_per_file = max(self.seg_num // max(files_count, 1), 10)  # 至少 10 個
        
        n_generated = 0
        max_attempts = 100000
        attempts = 0
        
        print(f"開始生成 {n_per_file} 個 patches...")
        
        while n_generated < n_per_file and attempts < max_attempts:
            attempts += 1
            
            # 隨機選擇一個點
            idx = np.random.randint(0, n_points)
            z, y, x = point_list[idx]
            
            # 計算裁剪範圍
            z1 = max(0, z - self.seg_z)
            z2 = min(data.shape[0], z + self.seg_z)
            y1 = max(0, y - self.halfy)
            y2 = min(data.shape[1], y + self.halfy)
            x1 = max(0, x - self.halfx)
            x2 = min(data.shape[2], x + self.halfx)
            
            # 檢查是否有足夠的 Z 切片
            available_z = z2 - z1
            if available_z < self.seg_z * 2:  # 需要足夠的 Z 切片做間隔採樣
                continue
            
            # **關鍵：Z 軸間隔採樣（複製 MATLAB 行為）**
            # MATLAB: input_crop = data(x1:x2,y1:y2,z1+1:2:z2)
            #         gt_crop = data(x1:x2,y1:y2,z1:2:z2)
            
            # 計算實際可用的 Z 範圍
            max_z_start = z2 - self.seg_z * 2
            z_start = np.random.randint(z1, max(z1+1, max_z_start+1))
            
            input_z_indices = np.arange(z_start+1, z_start+1+self.seg_z*2, 2)  # 奇數切片
            gt_z_indices = np.arange(z_start, z_start+self.seg_z*2, 2)         # 偶數切片
            
            # 確保 Z 切片數量正確
            if len(input_z_indices) != self.seg_z or len(gt_z_indices) != self.seg_z:
                continue
            
            # 檢查 Z 索引是否在範圍內
            if (np.max(input_z_indices) >= data.shape[0] or 
                np.max(gt_z_indices) >= data.shape[0]):
                continue
            
            # 擷取 patches
            input_crop = data[input_z_indices, y1:y2, x1:x2]
            gt_crop = data[gt_z_indices, y1:y2, x1:x2]
            
            # 檢查 patch 大小
            if (input_crop.shape[1] < self.seg_y or input_crop.shape[2] < self.seg_x or
                gt_crop.shape[1] < self.seg_y or gt_crop.shape[2] < self.seg_x):
                continue
            
            # 調整到正確大小
            input_crop = input_crop[:self.seg_z, :self.seg_y, :self.seg_x]
            gt_crop = gt_crop[:self.seg_z, :self.seg_y, :self.seg_x]
            
            # 調整維度順序到 (Y, X, Z)
            input_crop = np.transpose(input_crop, (1, 2, 0))
            gt_crop = np.transpose(gt_crop, (1, 2, 0))
            
            # 品質檢查
            # 1. 動態範圍檢查
            p99 = np.percentile(input_crop, 99.9)
            p01 = np.percentile(input_crop, 0.1) + 1e-2
            if p01 > 0:
                active_range = p99 / p01
                if active_range < self.active_range_thresh:
                    continue
            
            # 2. 總訊號檢查
            sum_value = np.sum(input_crop)
            if sum_value < self.sum_thresh:
                continue
            
            # 生成增強版本（原始 + 翻轉 + 旋轉）
            augmented_pairs = self.generate_augmentations(input_crop, gt_crop)
            
            # 儲存所有增強版本
            for inp, gt in augmented_pairs:
                if n_generated >= n_per_file:
                    break
                
                # 轉換為 uint16 並儲存
                inp_uint16 = (np.clip(inp, 0, 1) * 65535).astype(np.uint16)
                gt_uint16 = (np.clip(gt, 0, 1) * 65535).astype(np.uint16)
                
                # 調整回 (Z, Y, X) 格式儲存
                inp_save = np.transpose(inp_uint16, (2, 0, 1))
                gt_save = np.transpose(gt_uint16, (2, 0, 1))
                
                tifffile.imwrite(
                    str(self.input_path / f'{n_generated+1:05d}.tif'),
                    inp_save
                )
                tifffile.imwrite(
                    str(self.gt_path / f'{n_generated+1:05d}.tif'),
                    gt_save
                )
                
                n_generated += 1
                if n_generated % 10 == 0:
                    print(f"生成 patch {n_generated}/{n_per_file}")
        
        print(f"完成處理，總共生成 {n_generated} 個 patches (嘗試 {attempts} 次)")
        return n_generated
    
    def generate_augmentations(self, input_crop, gt_crop):
        """
        生成所有增強版本
        複製 MATLAB 的增強策略
        """
        augmented_pairs = []
        
        # 1. 原始版本 + 可能的旋轉
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_rot = self.rotate_volume(input_crop, angle)
            gt_rot = self.rotate_volume(gt_crop, angle)
            augmented_pairs.append((inp_rot, gt_rot))
        else:
            augmented_pairs.append((input_crop.copy(), gt_crop.copy()))
        
        # 2. 上下翻轉 (flipud)
        inp_flip_ud = np.flip(input_crop, axis=0)
        gt_flip_ud = np.flip(gt_crop, axis=0)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_ud = self.rotate_volume(inp_flip_ud, angle)
            gt_flip_ud = self.rotate_volume(gt_flip_ud, angle)
        augmented_pairs.append((inp_flip_ud, gt_flip_ud))
        
        # 3. 左右翻轉 (fliplr)
        inp_flip_lr = np.flip(input_crop, axis=1)
        gt_flip_lr = np.flip(gt_crop, axis=1)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_lr = self.rotate_volume(inp_flip_lr, angle)
            gt_flip_lr = self.rotate_volume(gt_flip_lr, angle)
        augmented_pairs.append((inp_flip_lr, gt_flip_lr))
        
        return augmented_pairs
    
    def rotate_volume(self, volume, angle):
        """
        旋轉 3D volume（逐層旋轉）
        複製 MATLAB imrotate 行為
        """
        if not self.rot_flag:
            return volume
        
        rotated = np.zeros((self.seg_x, self.seg_y, self.seg_z))
        
        # 創建較大的 canvas 供旋轉
        canvas_size = max(int(self.seg_x * 1.5), int(self.seg_y * 1.5))
        
        for z in range(volume.shape[2]):
            # 在較大的 canvas 上放置影像
            canvas = np.zeros((canvas_size, canvas_size))
            start_y = (canvas_size - volume.shape[0]) // 2
            start_x = (canvas_size - volume.shape[1]) // 2
            canvas[start_y:start_y+volume.shape[0], start_x:start_x+volume.shape[1]] = volume[:, :, z]
            
            # 旋轉
            slice_rot = rotate(canvas, angle, reshape=False, order=1, mode='constant', cval=0)
            
            # 裁剪到目標大小
            center_y = canvas_size // 2
            center_x = canvas_size // 2
            crop_y1 = center_y - self.seg_y // 2
            crop_y2 = crop_y1 + self.seg_y
            crop_x1 = center_x - self.seg_x // 2
            crop_x2 = crop_x1 + self.seg_x
            
            rotated[:, :, z] = slice_rot[crop_y1:crop_y2, crop_x1:crop_x2]
        
        return rotated
    
    def run(self):
        """執行資料增強"""
        print("=" * 60)
        print("開始執行 Python 版 DataAugmFor3D")
        print("=" * 60)
        print(f"輸入路徑: {self.data_path}")
        print(f"輸出路徑: {self.save_training_path}")
        print(f"Patch 大小: {self.seg_x}x{self.seg_y}x{self.seg_z}")
        print(f"目標數量: {self.seg_num}")
        print(f"旋轉增強: {'啟用' if self.rot_flag else '停用'}")
        print()
        
        # 獲取所有檔案
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob('*.tif')) + \
                   list(self.data_path.glob('*.tiff'))
        
        if not files:
            print("錯誤：沒有找到 TIFF 檔案")
            return
        
        print(f"找到 {len(files)} 個檔案:")
        for f in files:
            print(f"  - {f.name}")
        print()
        
        total_generated = 0
        for i, file in enumerate(files):
            print(f"[{i+1}/{len(files)}] 處理檔案: {file.name}")
            n = self.process_file(file)
            total_generated += n
            print(f"從 {file.name} 生成了 {n} 個 patches")
            print()
        
        print("=" * 60)
        print("資料增強完成！")
        print("=" * 60)
        print(f"總共生成: {total_generated} 個訓練樣本")
        print(f"Input 資料夾: {self.input_path}")
        print(f"GT 資料夾: {self.gt_path}")
        print()
        
        # 檢查輸出
        input_files = list(self.input_path.glob('*.tif'))
        gt_files = list(self.gt_path.glob('*.tif'))
        print(f"驗證: Input 檔案 {len(input_files)} 個, GT 檔案 {len(gt_files)} 個")
        
        if len(input_files) > 0:
            # 檢查第一個檔案
            sample = tifffile.imread(str(input_files[0]))
            print(f"樣本檔案形狀: {sample.shape}")
        
        return str(self.save_training_path)


# 使用範例
if __name__ == "__main__":
    # 設定參數（與 MATLAB 版本對應）
    augmentor = DataAugmFor3D(
        data_path='/home/aero/charliechang/projects/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python/data/ori_input/iUExM/iUExM_roi.tif',
        save_path='../your_augmented_datasets/iUExM/',
        seg_x=64,
        seg_y=64,
        seg_z=13,
        seg_num=100,
        rot_flag=True
    )
    
    # 執行
    augmentor.run()