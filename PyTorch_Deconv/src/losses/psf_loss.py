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
            y_pred: Predicted output from network (shape: B, C, H+2*insert_xy, W+2*insert_xy for upsample_flag=True: H*2, W*2)
            y_true: Ground truth target (shape: B, C, H_target, W_target)
            
        Returns:
            Total loss value
        """
        batch_size, channels, height, width = y_pred.shape
        
        # CRITICAL: Test different PSF processing approaches
        # Analysis of Fiji's permute_dimensions([1, 2, 3, 0]):
        # 
        # In TensorFlow/Keras:
        # - PSF likely comes in as (height, width, in_channels, out_channels)
        # - permute_dimensions([1, 2, 3, 0]) rearranges to (width, in_channels, out_channels, height)
        # - But this seems wrong for conv2d... Let me test different interpretations
        
        # Method 1: Original PyTorch approach (direct)
        y_conv_original = F.conv2d(y_pred, self.psf, padding='same')
        
        # Method 2: Test PSF rotation (180 degrees) - common in deconvolution
        psf_rotated = torch.rot90(self.psf, k=2, dims=[-2, -1])  # 180-degree rotation
        y_conv_rotated = F.conv2d(y_pred, psf_rotated, padding='same')
        
        # Method 3: Test PSF flip (both dimensions)
        psf_flipped = torch.flip(self.psf, dims=[-2, -1])
        y_conv_flipped = F.conv2d(y_pred, psf_flipped, padding='same')
        
        # Will select the best method after comparing results
        # (Selection happens in debug block below)
        
        # DEBUG: Track tensor shapes and values at each step
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            print(f"\n=== PSF Loss Debug ===")
            print(f"Input y_pred shape: {y_pred.shape}, range: [{y_pred.min().item():.6f}, {y_pred.max().item():.6f}]")
            print(f"PSF shape: {self.psf.shape}, sum: {self.psf.sum().item():.6f}")
            print(f"After conv - Original: shape {y_conv_original.shape}, range [{y_conv_original.min().item():.6f}, {y_conv_original.max().item():.6f}]")
            print(f"After conv - Rotated: shape {y_conv_rotated.shape}, range [{y_conv_rotated.min().item():.6f}, {y_conv_rotated.max().item():.6f}]")
            
            # Compare differences
            diff_rot = torch.mean(torch.abs(y_conv_original - y_conv_rotated)).item()
            print(f"Difference Original vs Rotated: {diff_rot:.6f}")
            
        # Select method based on which gives larger response (non-zero)
        if y_conv_original.max() > y_conv_rotated.max():
            y_conv = y_conv_original
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print("Selected: Original PSF method")
        else:
            y_conv = y_conv_rotated  
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print("Selected: Rotated PSF method")
        
        # Handle upsampling: if enabled, the y_conv is 2x size, need to downsample
        if self.upsample_flag:
            # Test different downsampling methods to match TensorFlow exactly
            # Method 1: Bilinear interpolation (current approach)
            y_conv_bilinear = F.interpolate(y_conv, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # Method 2: Nearest neighbor (matches TensorFlow's resize with different method)
            y_conv_nearest = F.interpolate(y_conv, scale_factor=0.5, mode='nearest')
            
            # Use bilinear downsampling (most common)
            y_conv = y_conv_bilinear
            
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"After downsampling: shape {y_conv.shape}, range [{y_conv.min().item():.6f}, {y_conv.max().item():.6f}]")
        
        # EXACT TensorFlow implementation: use full insert_xy for cropping
        # TensorFlow: y_conv[:,insert_xy:shape[1]-insert_xy,insert_xy:shape[2]-insert_xy,:]
        if self.insert_xy > 0:
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"Before cropping: shape {y_conv.shape}")
                print(f"Cropping with insert_xy={self.insert_xy}: [{self.insert_xy}:{y_conv.shape[2]-self.insert_xy}, {self.insert_xy}:{y_conv.shape[3]-self.insert_xy}]")
            
            y_conv = y_conv[:, :, self.insert_xy:-self.insert_xy, self.insert_xy:-self.insert_xy]
            
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"After cropping: shape {y_conv.shape}, range [{y_conv.min().item():.6f}, {y_conv.max().item():.6f}]")
                
        # Resize y_conv to match y_true size for loss calculation  
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            print(f"Target y_true shape: {y_true.shape}, range: [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
            
        if y_conv.shape[-2:] != y_true.shape[-2:]:
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"Resizing y_conv from {y_conv.shape[-2:]} to {y_true.shape[-2:]}")
            y_conv = F.interpolate(y_conv, size=y_true.shape[-2:], mode='bilinear', align_corners=False)
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(f"After resize: shape {y_conv.shape}, range [{y_conv.min().item():.6f}, {y_conv.max().item():.6f}]")
        
        # Main PSF loss (reconstruction loss)
        if self.use_mse:
            psf_loss = F.mse_loss(y_conv, y_true)
        else:
            psf_loss = F.l1_loss(y_conv, y_true)
        
        total_loss = psf_loss
        tv_loss = torch.tensor(0.0, device=y_pred.device)
        hessian_loss = torch.tensor(0.0, device=y_pred.device)
        l1_loss = torch.tensor(0.0, device=y_pred.device)
        
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
        
        # Final loss computation and reporting
        if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
            print(f"\nFinal y_conv shape: {y_conv.shape}, range [{y_conv.min().item():.6f}, {y_conv.max().item():.6f}]")
            print(f"Final y_true shape: {y_true.shape}, range [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
            print(f"Loss Components: PSF={psf_loss.item():.6f}, Hessian={hessian_loss.item():.6f}")
            print(f"Total Loss: {total_loss.item():.6f}")
            print("=" * 80)
        
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
        
        # Match TensorFlow's l2_loss calculation: l2_loss = sum(x^2)/2, then /size = mean(x^2)/2
        # Fix: Use higher precision calculation to avoid numerical issues
        xx_loss = torch.sum(xx ** 2) / (2.0 * xx.numel())
        yy_loss = torch.sum(yy ** 2) / (2.0 * yy.numel())
        xy_loss = torch.sum(xy ** 2) / (2.0 * xy.numel())
        yx_loss = torch.sum(yx ** 2) / (2.0 * yx.numel())
        
        hessian_loss = xx_loss + yy_loss + xy_loss + yx_loss
        
        return hessian_loss
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Custom state_dict that excludes PSF tensor to avoid DDP synchronization issues."""
        # Get the default state dict
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # Remove PSF tensor from state dict as it's set during initialization
        psf_key = prefix + 'psf'
        if psf_key in state_dict:
            del state_dict[psf_key]
        
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict that handles missing PSF tensor."""
        # PSF tensor is not included in state_dict, so we need to handle it
        # Create a copy of the state dict to avoid modifying the original
        state_dict_copy = state_dict.copy()
        
        # Add PSF tensor if it's missing (it should be missing due to our custom state_dict)
        if 'psf' not in state_dict_copy and hasattr(self, 'psf'):
            # PSF is already set during initialization, no need to load it
            pass
        
        # Load the state dict with strict=False to handle missing PSF gracefully
        return super().load_state_dict(state_dict_copy, strict=False)


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
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Custom state_dict that excludes PSF tensor to avoid DDP synchronization issues."""
        return super().state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict that handles PSF-related parameters."""
        return super().load_state_dict(state_dict, strict=False)


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