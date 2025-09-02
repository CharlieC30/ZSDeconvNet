import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import torchvision.utils as vutils
import numpy as np
import tifffile
import os
from typing import Any, Optional, Dict, Union

from .two_stage_unet import TwoStageUNet
from ..losses.two_stage_loss import create_two_stage_loss
from ..utils.psf_utils import PSFProcessor


class TwoStageDeconvolutionLightningModule(pl.LightningModule):
    """
    Lightning module for two-stage (denoising + deconvolution) training.
    Exactly matches TensorFlow implementation behavior.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 loss_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any],
                 psf_path: str,
                 psf_config: Optional[Dict[str, Any]] = None,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 log_images: bool = True,
                 log_every_n_epochs: int = 10):
        """
        Initialize two-stage Lightning module.
        
        Args:
            model_config: Configuration for the two-stage U-Net model
            loss_config: Configuration for the loss function (including stage weights)
            optimizer_config: Configuration for the optimizer
            psf_path: Path to the PSF file
            psf_config: Configuration for PSF processing
            scheduler_config: Configuration for learning rate scheduler
            log_images: Whether to log images during training
            log_every_n_epochs: How often to log images
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize two-stage model (remove architecture from config)
        model_config_filtered = {k: v for k, v in model_config.items() if k != 'architecture'}
        self.model = TwoStageUNet(**model_config_filtered)
        
        # Initialize PSF processor
        self.psf_processor = PSFProcessor(psf_path)
        self.psf_config = psf_config or {
            'target_dx': 0.0313,
            'target_dy': 0.0313,
            'psf_dx': None,
            'psf_dy': None
        }
        
        # Store configs
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.log_images = log_images
        self.log_every_n_epochs = log_every_n_epochs
        
        # Loss function will be initialized in setup()
        self.loss_fn = None
        
        # Metrics tracking
        self.train_loss_history = []
        self.val_loss_history = []
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the model for training/validation/testing."""
        
        # Load and process PSF
        target_shape = (self.hparams.model_config['insert_xy'] * 4,
                       self.hparams.model_config['insert_xy'] * 4)
        
        psf_tensor = self.psf_processor.load_psf(
            target_shape=target_shape,
            **self.psf_config
        )
        
        # Move PSF to same device as model
        psf_tensor = psf_tensor.to(self.device)
        
        # Initialize two-stage loss function
        self.loss_fn = create_two_stage_loss(psf_tensor, self.loss_config)
        
        print(f"=== Two-Stage Setup ===")
        print(f"PSF shape: {psf_tensor.shape}")
        print(f"PSF sum: {psf_tensor.sum().item():.6f}")
        print(f"Denoise loss weight: {self.loss_config.get('denoise_loss_weight', 0.5)}")
        print(f"Deconv loss weight: {self.loss_config.get('deconv_loss_weight', 0.5)}")
        print(f"Learning rate: {self.optimizer_config.get('lr', 0.001)}")
        print(f"Architecture: Two-Stage (Denoising + Deconvolution)")
        print("=" * 25)
        
    def forward(self, x):
        """Forward pass through the two-stage model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for two-stage model."""
        input_imgs, target_imgs = batch
        
        # Forward pass - returns [denoised_output, deconvolved_output]
        predictions = self.forward(input_imgs)
        
        # Compute two-stage loss
        loss_dict = self.loss_fn(predictions, target_imgs)
        
        # Log comprehensive metrics
        self.log('train_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_denoise_loss', loss_dict['denoise_loss'], on_step=True, on_epoch=True)
        self.log('train_deconv_loss', loss_dict['deconv_loss'], on_step=True, on_epoch=True)
        self.log('train_psf_loss', loss_dict['deconv_psf_loss'], on_step=True, on_epoch=True)
        self.log('train_hessian_loss', loss_dict['deconv_hessian_loss'], on_step=True, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        # Monitor stage outputs and gradients every 50 batches
        if batch_idx % 50 == 0:
            denoised, deconvolved = predictions
            self.log('stage1_mean', denoised.mean().item(), on_step=True)
            self.log('stage2_mean', deconvolved.mean().item(), on_step=True)
            self.log('stage1_std', denoised.std().item(), on_step=True)
            self.log('stage2_std', deconvolved.std().item(), on_step=True)
            
            # Monitor gradient magnitudes for Stage 1 and Stage 2
            stage1_grad_norm = 0.0
            stage2_grad_norm = 0.0
            stage1_param_count = 0
            stage2_param_count = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    if 'stage1_' in name:
                        stage1_grad_norm += param_grad_norm ** 2
                        stage1_param_count += 1
                    elif 'stage2_' in name:
                        stage2_grad_norm += param_grad_norm ** 2 
                        stage2_param_count += 1
            
            if stage1_param_count > 0:
                stage1_grad_norm = (stage1_grad_norm / stage1_param_count) ** 0.5
                self.log('stage1_grad_norm', stage1_grad_norm, on_step=True)
            
            if stage2_param_count > 0:
                stage2_grad_norm = (stage2_grad_norm / stage2_param_count) ** 0.5
                self.log('stage2_grad_norm', stage2_grad_norm, on_step=True)
            
            # Monitor Stage 1 output layer weights
            if hasattr(self.model, 'stage1_output'):
                stage1_weight_norm = self.model.stage1_output.weight.norm().item()
                stage1_bias_norm = self.model.stage1_output.bias.norm().item()
                self.log('stage1_weight_norm', stage1_weight_norm, on_step=True)
                self.log('stage1_bias_norm', stage1_bias_norm, on_step=True)
        
        # Store loss for history
        self.train_loss_history.append(loss_dict['total_loss'].item())
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step for two-stage model."""
        input_imgs, target_imgs = batch
        
        # Forward pass
        predictions = self.forward(input_imgs)
        
        # Compute two-stage loss
        loss_dict = self.loss_fn(predictions, target_imgs)
        
        # Log comprehensive metrics
        self.log('val_loss', loss_dict['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_denoise_loss', loss_dict['denoise_loss'], on_step=False, on_epoch=True)
        self.log('val_deconv_loss', loss_dict['deconv_loss'], on_step=False, on_epoch=True)
        self.log('val_psf_loss', loss_dict['deconv_psf_loss'], on_step=False, on_epoch=True)
        self.log('val_hessian_loss', loss_dict['deconv_hessian_loss'], on_step=False, on_epoch=True)
        
        # Store loss for history
        self.val_loss_history.append(loss_dict['total_loss'].item())
        
        # Log images occasionally
        if (self.log_images and 
            batch_idx == 0 and 
            self.current_epoch % self.log_every_n_epochs == 0):
            self._log_two_stage_images(input_imgs, target_imgs, predictions, 'val')
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        
        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **self.optimizer_config
        )
        
        # Setup scheduler if configured
        if self.scheduler_config:
            scheduler_type = self.scheduler_config.get('type', 'step')
            scheduler_params = {k: v for k, v in self.scheduler_config.items() if k != 'type'}
            
            if scheduler_type == 'step':
                scheduler = StepLR(optimizer, **scheduler_params)
            elif scheduler_type == 'exponential':
                scheduler = ExponentialLR(optimizer, **scheduler_params)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        
        return optimizer
    
    def _log_two_stage_images(self, inputs, targets, predictions, stage='val'):
        """Log sample images from two-stage model to tensorboard."""
        # predictions is [denoised_output, deconvolved_output]
        if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
            print(f"Warning: Expected 2 predictions, got {len(predictions) if isinstance(predictions, (list, tuple)) else 'non-list'}")
            return
            
        denoised_output, deconvolved_output = predictions
        
        # Take first image from batch
        input_img = inputs[0, 0].cpu()  # Remove batch and channel dims
        target_img = targets[0, 0].cpu()
        denoised_img = denoised_output[0, 0].cpu()
        deconv_img = deconvolved_output[0, 0].cpu()
        
        # Crop images to match target size
        target_h, target_w = target_img.shape
        
        # Crop input to match target size (remove padding)
        input_img = self._crop_to_target(input_img, target_h, target_w)
        denoised_img = self._crop_to_target(denoised_img, target_h, target_w)
        deconv_img = self._crop_to_target(deconv_img, target_h, target_w)
        
        # Normalize for visualization
        input_norm = self._normalize_for_vis(input_img)
        target_norm = self._normalize_for_vis(target_img)
        denoised_norm = self._normalize_for_vis(denoised_img)
        deconv_norm = self._normalize_for_vis(deconv_img)
        
        # Remove channel dimension and create comparison grid
        input_norm = input_norm.squeeze(0)
        target_norm = target_norm.squeeze(0)
        denoised_norm = denoised_norm.squeeze(0)
        deconv_norm = deconv_norm.squeeze(0)
        
        # Stack images for comparison: Input | Target | Denoised | Deconvolved
        comparison = torch.stack([input_norm, target_norm, denoised_norm, deconv_norm], dim=0)
        
        # Log to tensorboard
        self.logger.experiment.add_images(
            f'{stage}_two_stage_comparison',
            comparison,
            self.current_epoch,
            dataformats='CHW'
        )
        
        # Also log individual stages
        stage1_comparison = torch.stack([input_norm, target_norm, denoised_norm], dim=0)
        stage2_comparison = torch.stack([denoised_norm, target_norm, deconv_norm], dim=0)
        
        self.logger.experiment.add_images(
            f'{stage}_stage1_denoising',
            stage1_comparison,
            self.current_epoch,
            dataformats='CHW'
        )
        
        self.logger.experiment.add_images(
            f'{stage}_stage2_deconvolution',
            stage2_comparison,
            self.current_epoch,
            dataformats='CHW'
        )
    
    def _crop_to_target(self, img, target_h, target_w):
        """Crop image to target size."""
        if img.shape == (target_h, target_w):
            return img
            
        ih, iw = img.shape
        start_h = (ih - target_h) // 2
        start_w = (iw - target_w) // 2
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        # Handle cases where target is larger than source
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(ih, end_h)
        end_w = min(iw, end_w)
        
        return img[start_h:end_h, start_w:end_w]
    
    def _normalize_for_vis(self, img):
        """Normalize image for visualization."""
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img
        return img_norm.unsqueeze(0)  # Add channel dimension
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_loss_history:
            avg_train_loss = np.mean(self.train_loss_history[-len(self.trainer.train_dataloader):])
            self.log('avg_train_loss_epoch', avg_train_loss)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_loss_history:
            avg_val_loss = np.mean(self.val_loss_history[-len(self.trainer.val_dataloaders[0]):])
            self.log('avg_val_loss_epoch', avg_val_loss)
    
    def on_load_checkpoint(self, checkpoint):
        """Handle loading checkpoint with potential dynamic parameters."""
        state_dict = checkpoint.get('state_dict', {})
        
        # Handle stage2_output_bias that may be dynamically added during training
        if 'model.stage2_output_bias' in state_dict:
            # If bias exists in checkpoint but not in current model, add it
            if self.model.stage2_output_bias is None:
                bias_value = state_dict['model.stage2_output_bias']
                self.model.stage2_output_bias = nn.Parameter(bias_value.clone())
                print("Added stage2_output_bias parameter from checkpoint")
    
    def save_checkpoint_with_metadata(self, filepath):
        """Save checkpoint with additional metadata."""
        try:
            checkpoint = {
                'state_dict': self.state_dict(),
                'model_config': self.hparams.model_config,
                'loss_config': self.hparams.loss_config,
                'psf_path': self.hparams.psf_path,
                'psf_config': self.psf_config,
                'epoch': self.current_epoch,
                'train_loss_history': self.train_loss_history,
                'val_loss_history': self.val_loss_history
            }
            torch.save(checkpoint, filepath)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint with full metadata: {e}")
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'model_config': self.hparams.model_config,
                'loss_config': self.hparams.loss_config,
                'psf_path': self.hparams.psf_path,
                'psf_config': self.psf_config
            }
            torch.save(checkpoint, filepath)
            print(f"Saved fallback checkpoint to: {filepath}")
    
    @classmethod
    def load_from_checkpoint_with_metadata(cls, checkpoint_path, map_location=None):
        """Load model from checkpoint with metadata."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        model_config = checkpoint['model_config']
        loss_config = checkpoint['loss_config']
        psf_path = checkpoint['psf_path']
        psf_config = checkpoint.get('psf_config', {})
        
        model = cls(
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config={'lr': 1e-4},
            psf_path=psf_path,
            psf_config=psf_config
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.train_loss_history = checkpoint.get('train_loss_history', [])
        model.val_loss_history = checkpoint.get('val_loss_history', [])
        
        return model


def create_two_stage_lightning_module(config):
    """
    Factory function to create two-stage Lightning module from configuration.
    
    Args:
        config: Dictionary containing all configuration parameters
        
    Returns:
        TwoStageDeconvolutionLightningModule instance
    """
    return TwoStageDeconvolutionLightningModule(
        model_config=config['model'],
        loss_config=config['loss'], 
        optimizer_config=config['optimizer'],
        psf_path=config['psf_path'],
        psf_config=config.get('psf', {}),
        scheduler_config=config.get('scheduler'),
        log_images=config.get('log_images', True),
        log_every_n_epochs=config.get('log_every_n_epochs', 10)
    )