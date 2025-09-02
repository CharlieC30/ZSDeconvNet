import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorFlowCompatiblePSFLoss(nn.Module):
    """
    PSF loss that exactly matches TensorFlow implementation.
    Removes the inconsistent PSF rotation/flipping and uses direct convolution only.
    """
    
    def __init__(self, 
                 psf,
                 tv_weight=0.0,
                 hessian_weight=0.02,
                 l1_weight=0.0,
                 use_mse=False,
                 upsample_flag=True,
                 insert_xy=16):
        super().__init__()
        
        # Store PSF as parameter
        if isinstance(psf, torch.Tensor):
            self.register_buffer('psf', psf, persistent=False)
        else:
            self.register_buffer('psf', torch.tensor(psf, dtype=torch.float32), persistent=False)
            
        # Loss weights
        self.tv_weight = tv_weight
        self.hessian_weight = hessian_weight
        self.l1_weight = l1_weight
        self.use_mse = use_mse
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        
    def forward(self, y_pred, y_true):
        """
        Forward pass exactly matching TensorFlow create_psf_loss implementation.
        """
        batch_size, channels, height, width = y_pred.shape
        
        # Step 1: Re-blur with PSF (direct convolution, no rotation)
        # TensorFlow: y_conv = K.conv2d(y_pred, psf_local, padding='same')
        y_conv = F.conv2d(y_pred, self.psf, padding='same')
        
        # Step 2: Handle upsampling (if model outputs 2x resolution)
        # TensorFlow: if upsample_flag: y_conv = tf.image.resize(y_conv,[height//2,width//2])
        if self.upsample_flag:
            _, _, target_h, target_w = y_true.shape
            if height == 2 * target_h and width == 2 * target_w:
                y_conv = F.interpolate(y_conv, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Step 3: Apply cropping
        # TensorFlow: y_conv = y_conv[:,insert_xy:y_conv.shape[1]-insert_xy,insert_xy:y_conv.shape[2]-insert_xy,:]
        if self.insert_xy > 0:
            y_conv = y_conv[:, :, self.insert_xy:y_conv.shape[2]-self.insert_xy, 
                          self.insert_xy:y_conv.shape[3]-self.insert_xy]
        
        # Step 3.5: Ensure y_conv matches y_true size exactly
        if y_conv.shape[2:] != y_true.shape[2:]:
            y_conv = F.interpolate(y_conv, size=y_true.shape[2:], mode='bilinear', align_corners=False)
        
        # Step 4: PSF fidelity loss
        # TensorFlow: if mse_flag: psf_loss = K.mean(K.square(y_true - y_conv))
        #            else: psf_loss = K.mean(K.abs(y_true - y_conv))
        if self.use_mse:
            psf_loss = F.mse_loss(y_conv, y_true)
        else:
            psf_loss = F.l1_loss(y_conv, y_true)
        
        # Step 5: Total Variation loss
        tv_loss = torch.tensor(0.0, device=y_pred.device)
        if self.tv_weight > 0:
            # TensorFlow implementation of TV loss
            y_diff = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]
            x_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]
            tv_loss = (torch.mean(x_diff ** 2) + torch.mean(y_diff ** 2))
        
        # Step 6: Hessian loss (critical for point convergence!)
        hessian_loss = torch.tensor(0.0, device=y_pred.device)
        if self.hessian_weight > 0:
            # TensorFlow Hessian implementation
            # Compute first-order derivatives
            y_diff = y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]  # dy
            x_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]  # dx
            
            # Compute second-order derivatives (Hessian)
            xx = x_diff[:, :, :, :-1] - x_diff[:, :, :, 1:]      # d²/dx²
            yy = y_diff[:, :, :-1, :] - y_diff[:, :, 1:, :]      # d²/dy²
            xy = y_diff[:, :, :, :-1] - y_diff[:, :, :, 1:]      # d²/dxdy
            yx = x_diff[:, :, :-1, :] - x_diff[:, :, 1:, :]      # d²/dydx
            
            # Hessian loss (sum of squared second derivatives)
            hessian_loss = (torch.mean(xx ** 2) + torch.mean(yy ** 2) + 
                          torch.mean(xy ** 2) + torch.mean(yx ** 2))
        
        # Step 7: L1 sparsity loss
        l1_loss = torch.tensor(0.0, device=y_pred.device)
        if self.l1_weight > 0:
            # TensorFlow: l1_loss = K.mean(K.abs(y_pred))
            l1_loss = torch.mean(torch.abs(y_pred))
        
        # Step 8: Total loss combination
        # TensorFlow: return psf_loss + TV_weight*TV_loss + Hess_weight*Hess_loss + l1_rate*l1_loss
        total_loss = (psf_loss + 
                     self.tv_weight * tv_loss + 
                     self.hessian_weight * hessian_loss + 
                     self.l1_weight * l1_loss)
        
        return {
            'total_loss': total_loss,
            'psf_loss': psf_loss,
            'tv_loss': tv_loss,
            'hessian_loss': hessian_loss,
            'l1_loss': l1_loss
        }
    
    def update_psf(self, new_psf):
        """Update PSF tensor."""
        if isinstance(new_psf, torch.Tensor):
            self.psf.copy_(new_psf)
        else:
            self.psf.copy_(torch.tensor(new_psf, dtype=torch.float32, device=self.psf.device))


# Update the PSFLoss alias to use the new implementation
PSFLoss = TensorFlowCompatiblePSFLoss


def create_tensorflow_compatible_psf_loss(psf, config):
    """Factory function for creating TensorFlow-compatible PSF loss."""
    return TensorFlowCompatiblePSFLoss(
        psf=psf,
        tv_weight=config.get('tv_weight', 0.0),
        hessian_weight=config.get('hessian_weight', 0.02),
        l1_weight=config.get('l1_weight', 0.0),
        use_mse=config.get('use_mse', False),
        upsample_flag=config.get('upsample_flag', True),
        insert_xy=config.get('insert_xy', 16)
    )