"""
Loss functions and PSF utilities for ZS-DeconvNet PyTorch implementation
Combines loss functions with PSF/OTF processing utilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import cv2
import math


# ==============================================================================
# PSF and OTF Utilities (equivalent to original loss.py PSF functions)
# ==============================================================================

def gaussian_1d(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """
    1D Gaussian function
    
    Args:
        x: Input coordinates
        amplitude: Amplitude
        mean: Mean value
        sigma: Standard deviation
    
    Returns:
        Gaussian function values
    """
    return amplitude * np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.)))


def psf_estimator_3d(psf: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate 3D PSF sigma values (equivalent to original psf_estimator_3d)
    
    Args:
        psf: 3D PSF array (Z, Y, X)
    
    Returns:
        (sigma_y, sigma_x, sigma_z)
    """
    from .utils import prctile_norm
    
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_z = max_index[0][0]
    index_y = max_index[1][0] 
    index_x = max_index[2][0]
    
    # Estimate Y sigma
    y_coords = np.arange(shape[1])
    y_profile = prctile_norm(np.squeeze(psf[index_z, :, index_x]))
    try:
        fit_y, _ = curve_fit(gaussian_1d, y_coords, y_profile, p0=[1, index_y, 2])
        sigma_y = abs(fit_y[2])
        print(f'Estimated PSF sigma_y: {sigma_y}')
    except:
        sigma_y = 2.0
    
    # Estimate X sigma
    x_coords = np.arange(shape[2])
    x_profile = prctile_norm(np.squeeze(psf[index_z, index_y, :]))
    try:
        fit_x, _ = curve_fit(gaussian_1d, x_coords, x_profile, p0=[1, index_x, 2])
        sigma_x = abs(fit_x[2])
        print(f'Estimated PSF sigma_x: {sigma_x}')
    except:
        sigma_x = 2.0
    
    # Estimate Z sigma
    z_coords = np.arange(shape[0])
    z_profile = prctile_norm(np.squeeze(psf[:, index_y, index_x]))
    try:
        fit_z, _ = curve_fit(gaussian_1d, z_coords, z_profile, p0=[1, index_z, 2])
        sigma_z = abs(fit_z[2])
        print(f'Estimated PSF sigma_z: {sigma_z}')
    except:
        sigma_z = 2.0
    
    return sigma_y, sigma_x, sigma_z


def process_psf_complete(psf: np.ndarray,
                        dxpsf: float, dzpsf: float,
                        dx: float, dz: float,
                        input_x: int, input_y: int, input_z: int,
                        insert_xy: int, insert_z: int,
                        upsample_flag: int = 0) -> np.ndarray:
    """
    Complete PSF processing matching original TensorFlow implementation (lines 163-294)
    
    Args:
        psf: Original PSF (Y, X, Z) - matches original psf_g shape after transpose
        dxpsf, dzpsf: Original PSF pixel spacing
        dx, dz: Target pixel spacing
        input_x, input_y, input_z: Input dimensions
        insert_xy, insert_z: Padding values
        upsample_flag: Upsampling flag
    
    Returns:
        Processed PSF ready for convolution
    """
    import imageio
    
    psf_g = psf.transpose([1, 2, 0])  # (Y, X, Z) -> (X, Y, Z)
    psf_width, psf_height, psf_depth = psf_g.shape
    half_psf_depth = math.floor(psf_depth / 2)
    
    if psf_depth % 2 == 0:
        raise ValueError('The depth of PSF should be an odd number.')
    
    # Z-direction interpolation (matching original lines 172-194)
    z = np.arange((half_psf_depth + 1) * dzpsf, (psf_depth + 0.1) * dzpsf, dzpsf)
    zi = np.arange((half_psf_depth + 1) * dzpsf, (psf_depth + 0.1) * dzpsf, dz)
    if zi[-1] > z[-1]:
        zi = zi[0:-1]
    
    PSF1 = np.zeros((psf_width, psf_height, len(zi)))
    for i in range(psf_width):
        for j in range(psf_height):
            curCol = psf_g[i, j, half_psf_depth:psf_depth]
            interp = interp1d(z, curCol, 'slinear')
            PSF1[i, j, :] = interp(zi)
    
    z2 = np.zeros(half_psf_depth)
    zi2 = np.zeros(len(zi) - 1)
    for n in range(half_psf_depth):
        z2[half_psf_depth - n - 1] = z[0] - dzpsf * (n + 1)
    for n in range(zi2.shape[0]):
        zi2[len(zi) - 1 - n - 1] = zi[0] - dz * (n + 1)
    
    PSF2 = np.zeros((psf_width, psf_height, len(zi2)))
    for i in range(psf_width):
        for j in range(psf_height):
            curCol = psf_g[i, j, 0:half_psf_depth]
            interp = interp1d(z2, curCol, 'slinear', fill_value='extrapolate')
            PSF2[i, j, :] = interp(zi2)
    
    psf_g = np.concatenate((PSF2, PSF1), axis=2)
    psf_g = psf_g / np.sum(psf_g)
    psf_width, psf_height, psf_depth = psf_g.shape
    half_psf_width = psf_width // 2
    
    # X/Y direction interpolation (matching original lines 199-249)
    if psf_width % 2 == 1:
        sr_ratio = dxpsf / dx
        sr_x = round(psf_width * sr_ratio)
        if sr_x % 2 == 0:
            sr_x = sr_x - 1 if sr_x > psf_width * sr_ratio else sr_x + 1
        sr_y = round(psf_height * sr_ratio)
        if sr_y % 2 == 0:
            sr_y = sr_y - 1 if sr_y > psf_height * sr_ratio else sr_y + 1
        
        psf_tmp = psf_g
        psf_g = np.zeros([sr_x, sr_y, psf_depth])
        for z in range(psf_g.shape[2]):
            psf_g[:, :, z] = cv2.resize(psf_tmp[:, :, z], (sr_y, sr_x))
    else:
        # Linear interpolation for even dimensions (matching original)
        x = np.arange((half_psf_width + 1) * dxpsf, (psf_width + 0.1) * dxpsf, dxpsf)
        xi = np.arange((half_psf_width + 1) * dxpsf, (psf_width + 0.1) * dxpsf, dx)
        if xi[-1] > x[-1]:
            xi = xi[0:-1]
        
        PSF1 = np.zeros((len(xi), psf_height, psf_depth))
        for i in range(psf_height):
            for j in range(psf_depth):
                curCol = psf_g[half_psf_width:psf_width, i, j]
                interp = interp1d(x, curCol, 'slinear')
                PSF1[:, i, j] = interp(xi)
        
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x)):
            x2[len(x) - n - 1] = x[0] - dxpsf * n
        for n in range(len(xi)):
            xi2[len(xi) - n - 1] = xi[0] - dx * n
        
        PSF2 = np.zeros((len(xi2), psf_height, psf_depth))
        for i in range(psf_height):
            for j in range(psf_depth):
                curCol = psf_g[1:half_psf_width + 1 + psf_width % 2, i, j]
                interp = interp1d(x2, curCol, 'slinear')
                PSF2[:, i, j] = interp(xi2)
        
        psf_g = np.concatenate((PSF2[:-1, :, :], PSF1), axis=0)
        psf_g = psf_g / np.sum(psf_g)
        psf_width, psf_height, psf_depth = psf_g.shape
        half_psf_height = psf_height // 2
        
        # Y-direction interpolation (matching original lines 246-249)
        x = np.arange((half_psf_height + 1) * dxpsf, (psf_height + 0.1) * dxpsf, dxpsf)
        xi = np.arange((half_psf_height + 1) * dxpsf, (psf_height + 0.1) * dxpsf, dx)
        if xi[-1] > x[-1]:
            xi = xi[0:-1]
        
        PSF1 = np.zeros((psf_width, len(xi), psf_depth))
        for i in range(psf_width):
            for j in range(psf_depth):
                curCol = psf_g[i, half_psf_height:psf_height, j]
                interp = interp1d(x, curCol, 'slinear')
                PSF1[i, :, j] = interp(xi)
        
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x)):
            x2[len(x) - n - 1] = x[0] - dxpsf * n
        for n in range(len(xi)):
            xi2[len(xi) - n - 1] = xi[0] - dx * n
        
        PSF2 = np.zeros((psf_width, len(xi2), psf_depth))
        for i in range(psf_width):
            for j in range(psf_depth):
                curCol = psf_g[i, 1:half_psf_height + 1 + psf_height % 2, j]
                interp = interp1d(x2, curCol, 'slinear')
                PSF2[i, :, j] = interp(xi2)
        
        psf_g = np.concatenate((PSF2[:, :-1, :], PSF1), axis=1)
    
    psf_g = psf_g / np.sum(psf_g)
    
    # Crop PSF for faster computation (matching original lines 285-294)
    psf_width, psf_height, psf_depth = psf_g.shape
    halfz = min(psf_depth // 2, input_z - 1)
    psf_g = psf_g[:, :, psf_depth // 2 - halfz:psf_depth // 2 + halfz + 1]
    
    sigma_y, sigma_x, _ = psf_estimator_3d(psf_g.transpose([2, 1, 0]))
    ksize = int(sigma_y * 4)
    halfx = psf_width // 2
    halfy = psf_height // 2
    
    if ksize <= halfx:
        psf_g = psf_g[halfx - ksize:halfx + ksize + 1, halfy - ksize:halfy + ksize + 1, :]
    
    # Reshape for convolution (matching original reshape)
    psf_g = np.reshape(psf_g, (psf_g.shape[0], psf_g.shape[1], psf_g.shape[2], 1, 1)).astype(np.float32)
    psf_g = psf_g / np.sum(psf_g)
    
    return psf_g


def interpolate_psf_3d(psf: np.ndarray, 
                       dx_psf: float, dz_psf: float,
                       dx_target: float, dz_target: float) -> np.ndarray:
    """
    Simplified PSF interpolation (legacy compatibility)
    For full processing, use process_psf_complete
    """
    # This is a simplified version for backwards compatibility
    # For complete processing, use process_psf_complete
    return process_psf_complete(
        psf.transpose([1, 2, 0]), dx_psf, dz_psf, dx_target, dz_target,
        input_x=64, input_y=64, input_z=13, insert_xy=8, insert_z=2
    )


def prepare_psf_for_conv3d(psf: np.ndarray) -> torch.Tensor:
    """
    Prepare PSF for PyTorch 3D convolution
    
    Args:
        psf: PSF array (Z, Y, X)
    
    Returns:
        PSF tensor for conv3d (1, 1, D, H, W)
    """
    # Normalize PSF
    psf = psf / np.sum(psf)
    
    # Convert to PyTorch tensor and add batch/channel dimensions
    psf_tensor = torch.from_numpy(psf).float()
    psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    return psf_tensor


# ==============================================================================
# Loss Functions (PyTorch implementations)
# ==============================================================================

class NBR2NBRLoss(nn.Module):
    """
    NBR2NBR loss matching original TensorFlow implementation
    Includes main loss and regularization loss with pseudo GT
    """
    
    def __init__(self, mse_flag: int = 0, tv_rate: float = 0.0):
        super().__init__()
        self.mse_flag = mse_flag
        self.tv_rate = tv_rate
    
    def forward(self, gt_combined: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        NBR2NBR loss computation matching original TensorFlow
        
        Args:
            gt_combined: Combined tensor with shape [..., 2] where [..., 0] is GT and [..., 1] is output_G
            output: Model output
        
        Returns:
            Combined NBR2NBR loss
        """
        # Extract GT and output_G (matching original)
        gt = gt_combined[..., 0]  # gt
        output_G = gt_combined[..., 1]  # output_G
        
        # Ensure output has correct shape (remove channel dim if needed)
        if output.dim() == 5 and output.shape[1] == 1:
            output = output.squeeze(1)
        
        # Handle upsampling if output is larger than gt
        if output.shape[-3:] != gt.shape[-3:]:
            # Resize output to match gt size (matching original tf.image.resize logic)
            output = F.interpolate(output.unsqueeze(1), size=gt.shape[-3:], mode='trilinear', align_corners=True).squeeze(1)
        
        # Main loss (matching original loss = K.mean(K.abs(gt-output)))
        if self.mse_flag:
            main_loss = F.mse_loss(output, gt)
            reg_loss = F.mse_loss(output - gt, output_G)
        else:
            main_loss = F.l1_loss(output, gt)
            reg_loss = F.l1_loss(output - gt, output_G)
        
        # TV loss (matching original TV regularization)
        tv_loss = 0.0
        if self.tv_rate > 0:
            _, height, width, depth = output.shape
            
            # Compute finite differences (matching original tf.slice operations)
            if height > 1:
                y_diff = output[:, 1:, :, :] - output[:, :-1, :, :]
                tv_loss += torch.mean(y_diff ** 2)
            
            if width > 1:
                x_diff = output[:, :, 1:, :] - output[:, :, :-1, :]
                tv_loss += torch.mean(x_diff ** 2)
            
            if depth > 1:
                z_diff = output[:, :, :, 1:] - output[:, :, :, :-1]
                tv_loss += torch.mean(z_diff ** 2)
        
        # Combine losses (matching original return loss+reg_loss+TV_rate*TV_loss)
        total_loss = main_loss + reg_loss + self.tv_rate * tv_loss
        
        return total_loss


class PSFLoss3D(nn.Module):
    """
    3D PSF convolution loss for physics-informed training
    """
    
    def __init__(self, 
                 psf: np.ndarray,
                 mse_flag: int = 0,
                 upsample_flag: bool = False,
                 tv_weight: float = 0.0,
                 hess_weight: float = 0.1,
                 insert_xy: int = 8,
                 insert_z: int = 2):
        super().__init__()
        
        self.mse_flag = mse_flag
        self.upsample_flag = upsample_flag
        self.tv_weight = tv_weight
        self.hess_weight = hess_weight
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        
        # Convert PSF to PyTorch parameter (non-trainable)
        psf_tensor = prepare_psf_for_conv3d(psf)
        self.register_buffer('psf', psf_tensor)
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Apply PSF convolution and compute loss
        
        Args:
            y_true: Ground truth tensor (B, C, D, H, W)
            y_pred: Predicted tensor (B, C, D, H, W)
        
        Returns:
            Combined loss value
        """
        # Apply PSF convolution
        y_pred_conv = F.conv3d(y_pred, self.psf, padding='same')
        
        # Crop according to insert parameters
        if self.upsample_flag:
            insert_xy_local = self.insert_xy * 2
        else:
            insert_xy_local = self.insert_xy
        
        h, w, d = y_pred.shape[-3:]
        y_pred_conv = y_pred_conv[..., 
                                  self.insert_z:d - self.insert_z,
                                  insert_xy_local:h - insert_xy_local,
                                  insert_xy_local:w - insert_xy_local]
        
        # Resize if upsampling
        if self.upsample_flag:
            y_pred_conv = F.interpolate(y_pred_conv, 
                                       size=(y_pred_conv.shape[-3],
                                            y_pred_conv.shape[-2] // 2,
                                            y_pred_conv.shape[-1] // 2),
                                       mode='trilinear')
        
        # Main reconstruction loss
        if self.mse_flag:
            recon_loss = F.mse_loss(y_pred_conv, y_true)
        else:
            recon_loss = F.l1_loss(y_pred_conv, y_true)
        
        # Regularization terms
        total_loss = recon_loss
        
        # TV loss
        if self.tv_weight > 0:
            tv_loss = self._tv_loss(y_pred)
            total_loss += self.tv_weight * tv_loss
        
        # Hessian loss
        if self.hess_weight > 0:
            hess_loss = self._hessian_loss(y_pred)
            total_loss += self.hess_weight * hess_loss
        
        return total_loss
    
    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total Variation loss"""
        batch_size, channels, depth, height, width = x.shape
        
        tv_d = torch.sum(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
        tv_h = torch.sum(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
        tv_w = torch.sum(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
        
        return (tv_d + tv_h + tv_w) / (batch_size * channels * depth * height * width)
    
    def _hessian_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Hessian regularization loss"""
        # Calculate gradients
        grad_x = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        grad_y = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        grad_z = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        
        # Calculate second derivatives (Hessian components)
        hess_xx = grad_x[:, :, :, :, 1:] - grad_x[:, :, :, :, :-1]
        hess_yy = grad_y[:, :, :, 1:, :] - grad_y[:, :, :, :-1, :]
        hess_zz = grad_z[:, :, 1:, :, :] - grad_z[:, :, :-1, :, :]
        
        return torch.mean(torch.square(hess_xx)) + \
               torch.mean(torch.square(hess_yy)) + \
               torch.mean(torch.square(hess_zz))


class PSFLoss3DNBR2NBR(nn.Module):
    """
    Combined PSF and NBR2NBR loss for two-stage training
    """
    
    def __init__(self, 
                 psf: np.ndarray,
                 mse_flag: int = 0,
                 upsample_flag: bool = False,
                 tv_weight: float = 0.0,
                 hess_weight: float = 0.1,
                 insert_xy: int = 8,
                 insert_z: int = 2):
        super().__init__()
        
        self.mse_flag = mse_flag
        self.upsample_flag = upsample_flag
        self.tv_weight = tv_weight
        self.hess_weight = hess_weight
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        
        # PSF should be already processed - convert to PyTorch tensor directly
        if psf.ndim == 5:  # Already processed (H, W, D, 1, 1) format
            psf_tensor = torch.from_numpy(np.squeeze(psf)).float()
            # Reshape from (H, W, D) to (1, 1, D, H, W) for conv3d
            psf_tensor = psf_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        else:
            # Simple 3D PSF (Z, Y, X) - convert directly
            psf_tensor = torch.from_numpy(psf).float()
            # Rearrange to (1, 1, D, H, W) for conv3d
            psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        self.register_buffer('psf', psf_tensor)
    
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        PSF loss computation matching original create_psf_loss_3D_NBR2NBR
        
        Args:
            y_true: Target with shape [..., 2] where [..., 0] is GT and [..., 1] is output_G
            y_pred: Predicted tensor (B, C, D, H, W)
        
        Returns:
            PSF loss
        """
        # Extract GT and output_G (matching original)
        gt = y_true[..., 0]  # y_true[:,:,:,:,0]
        output_G = y_true[..., 1]  # y_true[:,:,:,:,1]
        
        # Get dimensions
        h, w, d = y_pred.shape[-3:]
        
        # Determine insert values based on upsampling
        if self.upsample_flag:
            insert_xy_local = self.insert_xy * 2
        else:
            insert_xy_local = self.insert_xy
        
        # Apply PSF convolution (matching original K.conv3d)
        if y_pred.dim() == 4:  # Add channel dimension if needed
            y_pred_conv = F.conv3d(y_pred.unsqueeze(1), self.psf, padding='same')
        else:
            y_pred_conv = F.conv3d(y_pred, self.psf, padding='same')
        
        # Crop to valid region (matching original cropping)
        y_pred_cropped = y_pred.squeeze(1) if y_pred.dim() == 5 else y_pred
        y_pred_cropped = y_pred_cropped[
            :, 
            insert_xy_local:h-insert_xy_local,
            insert_xy_local:w-insert_xy_local,
            self.insert_z:d-self.insert_z
        ]
        
        y_pred_conv_cropped = y_pred_conv.squeeze(1) if y_pred_conv.dim() == 5 else y_pred_conv
        y_pred_conv_cropped = y_pred_conv_cropped[
            :,
            insert_xy_local:h-insert_xy_local,
            insert_xy_local:w-insert_xy_local, 
            self.insert_z:d-self.insert_z
        ]
        
        # Handle upsampling (matching original tf.image.resize)
        if self.upsample_flag:
            target_size = (y_pred_conv_cropped.shape[-3] // 2, y_pred_conv_cropped.shape[-2] // 2)
            y_pred_conv_cropped = F.interpolate(
                y_pred_conv_cropped.unsqueeze(1), 
                size=(y_pred_conv_cropped.shape[-3], target_size[0], target_size[1]),
                mode='trilinear', align_corners=True
            ).squeeze(1)
        
        # Main loss (matching original loss = K.mean(K.abs(y_true-y_pred_conv)))
        if self.mse_flag:
            main_loss = F.mse_loss(y_pred_conv_cropped, gt)
            reg_loss = F.mse_loss(y_pred_conv_cropped - gt, output_G)
        else:
            main_loss = F.l1_loss(y_pred_conv_cropped, gt)
            reg_loss = F.l1_loss(y_pred_conv_cropped - gt, output_G)
        
        # Get dimensions for regularization
        h_reg, w_reg, d_reg = y_pred_cropped.shape[-3:]
        
        # TV loss (matching original)
        tv_loss = 0.0
        if self.tv_weight > 0:
            if h_reg > 1:
                y_diff = y_pred_cropped[:, 1:, :, :] - y_pred_cropped[:, :-1, :, :]
                tv_loss += torch.mean(y_diff ** 2)
            
            if w_reg > 1:
                x_diff = y_pred_cropped[:, :, 1:, :] - y_pred_cropped[:, :, :-1, :]
                tv_loss += torch.mean(x_diff ** 2)
            
            if d_reg > 1:
                z_diff = y_pred_cropped[:, :, :, 1:] - y_pred_cropped[:, :, :, :-1]
                tv_loss += torch.mean(z_diff ** 2)
        
        # Hessian loss (matching original)
        hess_loss = 0.0
        if self.hess_weight > 0:
            if h_reg > 1:
                x_grad = y_pred_cropped[:, 1:, :, :] - y_pred_cropped[:, :-1, :, :]
                if w_reg > 1:
                    x_hess = x_grad[:, :, 1:, :] - x_grad[:, :, :-1, :]
                    hess_loss += torch.mean(x_hess ** 2)
                if d_reg > 1:
                    x_hess = x_grad[:, :, :, 1:] - x_grad[:, :, :, :-1]
                    hess_loss += torch.mean(x_hess ** 2)
            
            if w_reg > 1:
                y_grad = y_pred_cropped[:, :, 1:, :] - y_pred_cropped[:, :, :-1, :]
                if h_reg > 1:
                    y_hess = y_grad[:, 1:, :, :] - y_grad[:, :-1, :, :]
                    hess_loss += torch.mean(y_hess ** 2)
                if d_reg > 1:
                    y_hess = y_grad[:, :, :, 1:] - y_grad[:, :, :, :-1]
                    hess_loss += torch.mean(y_hess ** 2)
            
            if d_reg > 1:
                z_grad = y_pred_cropped[:, :, :, 1:] - y_pred_cropped[:, :, :, :-1]
                if h_reg > 1:
                    z_hess = z_grad[:, 1:, :, :] - z_grad[:, :-1, :, :]
                    hess_loss += torch.mean(z_hess ** 2)
                if w_reg > 1:
                    z_hess = z_grad[:, :, 1:, :] - z_grad[:, :, :-1, :]
                    hess_loss += torch.mean(z_hess ** 2)
        
        # Combine all losses (matching original)
        total_loss = main_loss + reg_loss + self.tv_weight * tv_loss + self.hess_weight * hess_loss
        
        return total_loss


# Factory functions for creating loss functions
def create_nbr2nbr_loss(tv_rate: float = 0.0, mse_flag: int = 0) -> NBR2NBRLoss:
    """Create NBR2NBR loss function matching original create_NBR2NBR_loss"""
    return NBR2NBRLoss(mse_flag, tv_rate)


def create_psf_loss_3d_nbr2nbr(psf: np.ndarray, 
                              mse_flag: int = 0,
                              batch_size: int = 1,
                              upsample_flag: bool = False,
                              tv_weight: float = 0.0,
                              hess_weight: float = 0.1,
                              insert_xy: int = 8,
                              insert_z: int = 2) -> PSFLoss3DNBR2NBR:
    """Create combined PSF and NBR2NBR loss function"""
    return PSFLoss3DNBR2NBR(
        psf=psf,
        mse_flag=mse_flag,
        upsample_flag=upsample_flag,
        tv_weight=tv_weight,
        hess_weight=hess_weight,
        insert_xy=insert_xy,
        insert_z=insert_z
    )


__all__ = [
    # PSF utilities
    'gaussian_1d',
    'psf_estimator_3d', 
    'interpolate_psf_3d',
    'prepare_psf_for_conv3d',
    
    # Loss classes
    'NBR2NBRLoss',
    'PSFLoss3D',
    'PSFLoss3DNBR2NBR',
    
    # Factory functions
    'create_nbr2nbr_loss',
    'create_psf_loss_3d_nbr2nbr'
]