import torch
import torch.nn as nn
import torch.nn.functional as F


class PSFConvolutionLoss(nn.Module):
    """
    PSF-based deconvolution loss function.
    Converted from TensorFlow version in the original codebase.
    """
    
    def __init__(self, 
                 psf_tensor,
                 tv_weight=0.0,
                 hessian_weight=0.02,
                 l1_weight=0.0,
                 use_mse=False,
                 upsample_flag=True,
                 insert_xy=16):
        """
        Initialize PSF convolution loss.
        
        Args:
            psf_tensor: PSF tensor of shape (1, 1, H, W)
            tv_weight: Weight for Total Variation regularization
            hessian_weight: Weight for Hessian regularization
            l1_weight: Weight for L1 regularization
            use_mse: If True, use MSE; otherwise use MAE
            upsample_flag: Whether the output is upsampled
            insert_xy: Padding size that needs to be cropped
        """
        super().__init__()
        
        self.register_buffer('psf', psf_tensor)
        self.tv_weight = tv_weight
        self.hessian_weight = hessian_weight
        self.l1_weight = l1_weight
        self.use_mse = use_mse
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        
    def forward(self, y_pred, y_true):
        """
        Compute PSF convolution loss.
        
        Args:
            y_pred: Predicted output from network
            y_true: Ground truth target
            
        Returns:
            Total loss value
        """
        batch_size, channels, height, width = y_pred.shape
        
        # Apply PSF convolution to prediction
        y_conv = F.conv2d(y_pred, self.psf, padding='same')
        
        # Handle upsampling if enabled
        if self.upsample_flag:
            y_conv = F.interpolate(y_conv, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Crop the convolved output to remove padding
        if self.insert_xy > 0:
            crop_start = self.insert_xy
            crop_end_h = y_conv.shape[2] - self.insert_xy
            crop_end_w = y_conv.shape[3] - self.insert_xy
            y_conv = y_conv[:, :, crop_start:crop_end_h, crop_start:crop_end_w]
        
        # Main PSF loss (reconstruction loss)
        if self.use_mse:
            psf_loss = F.mse_loss(y_conv, y_true)
        else:
            psf_loss = F.l1_loss(y_conv, y_true)
        
        total_loss = psf_loss
        
        # Total Variation regularization
        if self.tv_weight > 0:
            tv_loss = self._compute_tv_loss(y_pred)
            total_loss += self.tv_weight * tv_loss
        
        # Hessian regularization
        if self.hessian_weight > 0:
            hessian_loss = self._compute_hessian_loss(y_pred)
            total_loss += self.hessian_weight * hessian_loss
        
        # L1 regularization
        if self.l1_weight > 0:
            l1_loss = torch.mean(torch.abs(y_pred))
            total_loss += self.l1_weight * l1_loss
        
        return total_loss
    
    def _compute_tv_loss(self, y_pred):
        """Compute Total Variation loss."""
        # Differences along height
        y_diff = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]
        # Differences along width
        x_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]
        
        y_tv = torch.mean(y_diff ** 2)
        x_tv = torch.mean(x_diff ** 2)
        
        return x_tv + y_tv
    
    def _compute_hessian_loss(self, y_pred):
        """Compute Hessian regularization loss."""
        # First derivatives
        y_diff = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]
        x_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]
        
        # Second derivatives (Hessian components)
        xx = x_diff[:, :, :, :-1] - x_diff[:, :, :, 1:]
        yy = y_diff[:, :, :-1, :] - y_diff[:, :, 1:, :]
        xy = y_diff[:, :, :, :-1] - y_diff[:, :, :, 1:]
        yx = x_diff[:, :, :-1, :] - x_diff[:, :, 1:, :]
        
        hessian_loss = (torch.mean(xx ** 2) + torch.mean(yy ** 2) + 
                       torch.mean(xy ** 2) + torch.mean(yx ** 2))
        
        return hessian_loss


class DeconvolutionLoss(nn.Module):
    """
    Complete deconvolution loss combining PSF loss with regularization terms.
    """
    
    def __init__(self, psf_tensor, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'tv_weight': 0.0,
                'hessian_weight': 0.02,
                'l1_weight': 0.0,
                'use_mse': False,
                'upsample_flag': True,
                'insert_xy': 16
            }
        
        self.psf_loss = PSFConvolutionLoss(psf_tensor, **config)
        
    def forward(self, y_pred, y_true):
        return self.psf_loss(y_pred, y_true)
    
    def update_psf(self, new_psf_tensor):
        """Update the PSF tensor."""
        self.psf_loss.psf = new_psf_tensor


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for better training stability.
    """
    
    def __init__(self, psf_tensor, scales=[1.0, 0.5], weights=[1.0, 0.5], config=None):
        super().__init__()
        
        self.scales = scales
        self.weights = weights
        self.losses = nn.ModuleList()
        
        for scale in scales:
            if scale != 1.0:
                # Create scaled PSF
                scaled_psf = F.interpolate(psf_tensor, scale_factor=scale, 
                                         mode='bilinear', align_corners=False)
                # Renormalize
                scaled_psf = scaled_psf / torch.sum(scaled_psf)
            else:
                scaled_psf = psf_tensor
            
            loss_config = config.copy() if config else {}
            if scale != 1.0:
                loss_config['upsample_flag'] = False  # Disable upsampling for scaled versions
            
            self.losses.append(PSFConvolutionLoss(scaled_psf, **loss_config))
    
    def forward(self, y_pred, y_true):
        total_loss = 0
        
        for i, (scale, weight, loss_fn) in enumerate(zip(self.scales, self.weights, self.losses)):
            if scale != 1.0:
                # Scale prediction and target
                scaled_pred = F.interpolate(y_pred, scale_factor=scale, 
                                          mode='bilinear', align_corners=False)
                scaled_true = F.interpolate(y_true, scale_factor=scale, 
                                          mode='bilinear', align_corners=False)
            else:
                scaled_pred = y_pred
                scaled_true = y_true
            
            loss_value = loss_fn(scaled_pred, scaled_true)
            total_loss += weight * loss_value
        
        return total_loss


def create_loss_function(psf_tensor, loss_type='psf', config=None):
    """
    Factory function to create loss functions.
    
    Args:
        psf_tensor: PSF tensor
        loss_type: Type of loss ('psf', 'multiscale')
        config: Loss configuration dictionary
    
    Returns:
        Loss function
    """
    if loss_type == 'psf':
        return DeconvolutionLoss(psf_tensor, config)
    elif loss_type == 'multiscale':
        return MultiScaleLoss(psf_tensor, config=config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy PSF
    psf = torch.randn(1, 1, 21, 21)
    psf = psf / torch.sum(psf)
    psf = psf.to(device)
    
    # Create loss function
    loss_fn = create_loss_function(psf)
    loss_fn = loss_fn.to(device)
    
    # Test with dummy data
    y_pred = torch.randn(2, 1, 128, 128, device=device, requires_grad=True)
    y_true = torch.randn(2, 1, 96, 96, device=device)  # Smaller due to cropping
    
    loss = loss_fn(y_pred, y_true)
    print(f"Loss value: {loss.item()}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful")