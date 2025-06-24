import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import torchvision.utils as vutils
import numpy as np
import tifffile
import os
from typing import Any, Optional, Dict

from .deconv_unet import DeconvUNet
from ..losses.psf_loss import create_loss_function
from ..utils.psf_utils import PSFProcessor


class DeconvolutionLightningModule(pl.LightningModule):
    """
    Lightning module for deconvolution training.
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
        Initialize Lightning module.
        
        Args:
            model_config: Configuration for the U-Net model
            loss_config: Configuration for the loss function  
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
        
        # Initialize model
        self.model = DeconvUNet(**model_config)
        
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
        # Use a typical patch size for PSF processing
        target_shape = (self.hparams.model_config['insert_xy'] * 4,
                       self.hparams.model_config['insert_xy'] * 4)
        
        psf_tensor = self.psf_processor.load_psf(
            target_shape=target_shape,
            **self.psf_config
        )
        
        # Move PSF to same device as model
        psf_tensor = psf_tensor.to(self.device)
        
        # Initialize loss function
        self.loss_fn = create_loss_function(
            psf_tensor, 
            loss_type='psf',
            config=self.loss_config
        )
        
        print(f"PSF shape: {psf_tensor.shape}")
        print(f"PSF sum: {psf_tensor.sum().item():.6f}")
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_imgs, target_imgs = batch
        
        # Forward pass
        predictions = self.forward(input_imgs)
        
        # Compute loss
        loss = self.loss_fn(predictions, target_imgs)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        # Store loss for history
        self.train_loss_history.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_imgs, target_imgs = batch
        
        # Forward pass
        predictions = self.forward(input_imgs)
        
        # Compute loss
        val_loss = self.loss_fn(predictions, target_imgs)
        
        # Log metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store loss for history
        self.val_loss_history.append(val_loss.item())
        
        # Log images occasionally
        if (self.log_images and 
            batch_idx == 0 and 
            self.current_epoch % self.log_every_n_epochs == 0):
            self._log_images(input_imgs, target_imgs, predictions, 'val')
        
        return val_loss
    
    
    
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
    
    def _log_images(self, inputs, targets, predictions, stage='val'):
        """Log sample images to tensorboard."""
        # Take first image from batch
        input_img = inputs[0, 0].cpu()  # Remove batch and channel dims
        target_img = targets[0, 0].cpu()
        pred_img = predictions[0, 0].cpu()
        
        # Crop input and prediction to match target size
        target_h, target_w = target_img.shape
        
        # Crop input to match target size (remove padding)
        if input_img.shape != target_img.shape:
            ih, iw = input_img.shape
            start_h = (ih - target_h) // 2
            start_w = (iw - target_w) // 2
            input_img = input_img[start_h:start_h + target_h, start_w:start_w + target_w]
        
        # Crop prediction to match target size (in case of upsampling)
        if pred_img.shape != target_img.shape:
            ph, pw = pred_img.shape
            start_h = (ph - target_h) // 2
            start_w = (pw - target_w) // 2
            pred_img = pred_img[start_h:start_h + target_h, start_w:start_w + target_w]
        
        # Normalize for visualization
        input_norm = self._normalize_for_vis(input_img)
        target_norm = self._normalize_for_vis(target_img)
        pred_norm = self._normalize_for_vis(pred_img)
        
        # Remove channel dimension and create comparison grid
        input_norm = input_norm.squeeze(0)  # Remove channel dim: (1, H, W) -> (H, W)
        target_norm = target_norm.squeeze(0)
        pred_norm = pred_norm.squeeze(0)
        
        # Stack images: (3, H, W)
        comparison = torch.stack([input_norm, target_norm, pred_norm], dim=0)
        
        # Log to tensorboard
        self.logger.experiment.add_images(
            f'{stage}_comparison',
            comparison,
            self.current_epoch,
            dataformats='CHW'
        )
    
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
        # Log average training loss
        if self.train_loss_history:
            avg_train_loss = np.mean(self.train_loss_history[-len(self.trainer.train_dataloader):])
            self.log('avg_train_loss_epoch', avg_train_loss)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Log average validation loss
        if self.val_loss_history:
            avg_val_loss = np.mean(self.val_loss_history[-len(self.trainer.val_dataloaders[0]):])
            self.log('avg_val_loss_epoch', avg_val_loss)
    
    def save_checkpoint_with_metadata(self, filepath):
        """Save checkpoint with additional metadata."""
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
        
    @classmethod
    def load_from_checkpoint_with_metadata(cls, checkpoint_path, map_location=None):
        """Load model from checkpoint with metadata."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract configurations
        model_config = checkpoint['model_config']
        loss_config = checkpoint['loss_config']
        psf_path = checkpoint['psf_path']
        psf_config = checkpoint.get('psf_config', {})
        
        # Create model
        model = cls(
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config={'lr': 1e-4},  # Default, will be overridden
            psf_path=psf_path,
            psf_config=psf_config
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        # Restore history
        model.train_loss_history = checkpoint.get('train_loss_history', [])
        model.val_loss_history = checkpoint.get('val_loss_history', [])
        
        return model


def create_lightning_module(config):
    """
    Factory function to create Lightning module from configuration.
    
    Args:
        config: Dictionary containing all configuration parameters
        
    Returns:
        DeconvolutionLightningModule instance
    """
    return DeconvolutionLightningModule(
        model_config=config['model'],
        loss_config=config['loss'], 
        optimizer_config=config['optimizer'],
        psf_path=config['psf_path'],
        psf_config=config.get('psf', {}),
        scheduler_config=config.get('scheduler'),
        log_images=config.get('log_images', True),
        log_every_n_epochs=config.get('log_every_n_epochs', 10)
    )


