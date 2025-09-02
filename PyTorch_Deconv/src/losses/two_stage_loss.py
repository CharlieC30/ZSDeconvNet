import torch
import torch.nn as nn
import torch.nn.functional as F
from .tensorflow_compatible_psf_loss import TensorFlowCompatiblePSFLoss


class TwoStageLoss(nn.Module):
    """
    Two-stage loss function for ZS-DeconvNet.
    Exactly matches TensorFlow implementation with:
    - Stage 1: Denoising loss (MAE/MSE)
    - Stage 2: Deconvolution loss (PSF + regularization)
    """
    
    def __init__(self, 
                 psf,
                 denoise_loss_weight=0.5,
                 deconv_loss_weight=0.5,
                 tv_weight=0.0,
                 hessian_weight=0.02,
                 l1_weight=0.0,
                 use_mse=False,
                 upsample_flag=True,
                 insert_xy=16):
        super().__init__()
        
        self.denoise_loss_weight = denoise_loss_weight
        self.deconv_loss_weight = deconv_loss_weight
        
        # Stage 1: Simple denoising loss
        if use_mse:
            self.denoise_loss_fn = nn.MSELoss()
        else:
            self.denoise_loss_fn = nn.L1Loss()
        
        # Stage 2: Complex deconvolution loss (use TensorFlow-compatible implementation)
        self.deconv_loss_fn = TensorFlowCompatiblePSFLoss(
            psf=psf,
            tv_weight=tv_weight,
            hessian_weight=hessian_weight,
            l1_weight=l1_weight,
            use_mse=use_mse,
            upsample_flag=upsample_flag,
            insert_xy=insert_xy
        )
    
    def forward(self, predictions, targets):
        """
        Forward pass for two-stage loss.
        
        Args:
            predictions: [denoised_output, deconvolved_output] from model
            targets: Ground truth tensor (same for both stages in TensorFlow)
        
        Returns:
            dict with loss components for logging
        """
        if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
            raise ValueError("TwoStageLoss expects predictions as [denoised_output, deconvolved_output]")
        
        denoised_output, deconvolved_output = predictions
        
        # Stage 1: Denoising loss
        # In TensorFlow: MAE/MSE between denoised output and ground truth
        denoise_loss = self.denoise_loss_fn(denoised_output, targets)
        
        # Stage 2: Deconvolution loss  
        # In TensorFlow: PSF loss + regularization
        deconv_loss_dict = self.deconv_loss_fn(deconvolved_output, targets)
        deconv_loss = deconv_loss_dict['total_loss']
        
        # Total loss: weighted combination (matches TensorFlow)
        total_loss = (self.denoise_loss_weight * denoise_loss + 
                     self.deconv_loss_weight * deconv_loss)
        
        # Return comprehensive loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'denoise_loss': denoise_loss,
            'deconv_loss': deconv_loss,
            'deconv_psf_loss': deconv_loss_dict.get('psf_loss', 0.0),
            'deconv_tv_loss': deconv_loss_dict.get('tv_loss', 0.0),
            'deconv_hessian_loss': deconv_loss_dict.get('hessian_loss', 0.0),
            'deconv_l1_loss': deconv_loss_dict.get('l1_loss', 0.0),
            'denoise_weight': self.denoise_loss_weight,
            'deconv_weight': self.deconv_loss_weight
        }
        
        return loss_dict
    
    def update_psf(self, new_psf):
        """Update PSF for deconvolution loss."""
        self.deconv_loss_fn.update_psf(new_psf)


def create_two_stage_loss(psf, config):
    """Factory function to create TwoStageLoss from configuration."""
    return TwoStageLoss(
        psf=psf,
        denoise_loss_weight=config.get('denoise_loss_weight', 0.5),
        deconv_loss_weight=config.get('deconv_loss_weight', 0.5),
        tv_weight=config.get('tv_weight', 0.0),
        hessian_weight=config.get('hessian_weight', 0.02),
        l1_weight=config.get('l1_weight', 0.0),
        use_mse=config.get('use_mse', False),
        upsample_flag=config.get('upsample_flag', True),
        insert_xy=config.get('insert_xy', 16)
    )