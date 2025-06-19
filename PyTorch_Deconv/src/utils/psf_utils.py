import torch
import numpy as np
import tifffile
import cv2
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import asarray as ar


def prctile_norm(x, min_prc=0, max_prc=100):
    """Percentile normalization."""
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def gaussian_1d(x, *param):
    """1D Gaussian function for PSF estimation."""
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))


def psf_estimator_2d(psf):
    """Estimate PSF sigma parameters."""
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    
    # Estimate y sigma
    x = ar(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x]))
    try:
        fit_y, _ = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2])
        sigma_y = fit_y[2]
    except:
        sigma_y = 2.0  # Default fallback
    
    # Estimate x sigma
    x = ar(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :]))
    try:
        fit_x, _ = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2])
        sigma_x = fit_x[2]
    except:
        sigma_x = 2.0  # Default fallback
    
    return sigma_y, sigma_x


def load_and_process_psf(psf_path, target_shape, target_dx, target_dy, psf_dx=None, psf_dy=None):
    """
    Load and process PSF for deconvolution.
    
    Args:
        psf_path: Path to PSF TIFF file
        target_shape: Target shape (height, width) for PSF
        target_dx: Target sampling interval in x direction
        target_dy: Target sampling interval in y direction
        psf_dx: PSF sampling interval in x direction (if None, read from file)
        psf_dy: PSF sampling interval in y direction (if None, read from file)
    
    Returns:
        torch.Tensor: Processed PSF ready for convolution
    """
    # Load PSF
    psf = tifffile.imread(psf_path).astype(np.float32)
    
    # If 3D PSF, take middle slice
    if len(psf.shape) == 3:
        psf = psf[psf.shape[0] // 2]
    
    psf_height, psf_width = psf.shape
    
    # Handle PSF resampling if needed
    if psf_dx is not None and psf_dy is not None:
        if psf_dx != target_dx or psf_dy != target_dy:
            # Interpolate PSF to match target sampling
            sr_ratio_x = psf_dx / target_dx
            sr_ratio_y = psf_dy / target_dy
            
            new_width = int(psf_width * sr_ratio_x)
            new_height = int(psf_height * sr_ratio_y)
            
            # Ensure odd dimensions
            if new_width % 2 == 0:
                new_width += 1
            if new_height % 2 == 0:
                new_height += 1
            
            psf = cv2.resize(psf, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Resize PSF to target shape if necessary
    if psf.shape != target_shape:
        psf = cv2.resize(psf, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize PSF
    psf = psf / np.sum(psf)
    
    # Crop PSF for computational efficiency
    sigma_y, sigma_x = psf_estimator_2d(psf)
    ksize = int(max(sigma_y, sigma_x) * 4)
    
    half_h = psf.shape[0] // 2
    half_w = psf.shape[1] // 2
    
    if ksize <= min(half_h, half_w):
        # Crop PSF around center
        h_start = max(0, half_h - ksize)
        h_end = min(psf.shape[0], half_h + ksize + 1)
        w_start = max(0, half_w - ksize)
        w_end = min(psf.shape[1], half_w + ksize + 1)
        
        psf_cropped = psf[h_start:h_end, w_start:w_end]
    else:
        psf_cropped = psf
    
    # Renormalize after cropping
    psf_cropped = psf_cropped / np.sum(psf_cropped)
    
    # Convert to torch tensor and add batch/channel dimensions
    psf_tensor = torch.from_numpy(psf_cropped).float()
    psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    return psf_tensor


def save_psf_visualization(psf_tensor, save_path):
    """Save PSF visualization for debugging."""
    psf_np = psf_tensor.squeeze().cpu().numpy()
    psf_vis = (prctile_norm(psf_np) * 65535).astype(np.uint16)
    tifffile.imwrite(save_path, psf_vis)


class PSFProcessor:
    """Class to handle PSF loading and processing."""
    
    def __init__(self, psf_path, device='cpu'):
        self.psf_path = psf_path
        self.device = device
        self.psf_tensor = None
        
    def load_psf(self, target_shape, target_dx, target_dy, psf_dx=None, psf_dy=None):
        """Load and process PSF."""
        self.psf_tensor = load_and_process_psf(
            self.psf_path, target_shape, target_dx, target_dy, psf_dx, psf_dy
        )
        self.psf_tensor = self.psf_tensor.to(self.device)
        return self.psf_tensor
    
    def get_psf_tensor(self):
        """Get the processed PSF tensor."""
        if self.psf_tensor is None:
            raise ValueError("PSF not loaded. Call load_psf() first.")
        return self.psf_tensor
    
    def to(self, device):
        """Move PSF to specified device."""
        self.device = device
        if self.psf_tensor is not None:
            self.psf_tensor = self.psf_tensor.to(device)
        return self


if __name__ == "__main__":
    # Test PSF loading
    import os
    
    # Test with your PSF file
    psf_path = "/path/to/your/psf.tif"  # Update this path
    if os.path.exists(psf_path):
        target_shape = (64, 64)
        target_dx = 0.0313
        target_dy = 0.0313
        
        psf_tensor = load_and_process_psf(psf_path, target_shape, target_dx, target_dy)
        print(f"PSF tensor shape: {psf_tensor.shape}")
        print(f"PSF sum: {psf_tensor.sum().item()}")
    else:
        print("PSF file not found. Update the path for testing.")