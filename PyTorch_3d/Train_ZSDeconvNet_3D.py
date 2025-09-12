#!/usr/bin/env python3
"""
ZS-DeconvNet 3D Training Script - PyTorch Implementation
Equivalent to original TensorFlow Train_ZSDeconvNet_3D.py
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# Removed DataLoader import as we're using manual training loop
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import tifffile
import datetime
from typing import Tuple, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models import twostage_RCAN3D, twostage_Unet3D
from utils import (create_nbr2nbr_loss, create_psf_loss_3d_nbr2nbr, 
                  psf_estimator_3d, process_psf_complete, prepare_psf_for_conv3d,
                  prctile_norm, read_tiff_stack, save_tiff_stack, SimplifiedDataModule)


class ZSDeconvNet3DModule(pl.LightningModule):
    """
    PyTorch Lightning module for ZS-DeconvNet 3D training
    Matches original TensorFlow implementation with g, g_copy models
    """
    
    def __init__(self, 
                 model_name: str = "twostage_RCAN3D",
                 psf_path: str = "",
                 input_shape: Tuple[int, int, int] = (64, 64, 13),
                 upsample_flag: int = 0,
                 insert_xy: int = 8,
                 insert_z: int = 2,
                 dx: float = 0.0926,
                 dz: float = 0.3704,
                 dxpsf: float = 0.0926,
                 dzpsf: float = 0.05,
                 mse_flag: int = 0,
                 tv_weight: float = 0.0,
                 hess_weight: float = 0.1,
                 learning_rate: float = 1e-4,
                 lr_decay_factor: float = 0.5,
                 lr_decay_steps: list = [5000, 7500],
                 batch_size: int = 3):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.model_name = model_name
        self.input_shape = input_shape
        self.upsample_flag = upsample_flag
        self.insert_xy = insert_xy
        self.insert_z = insert_z
        self.batch_size = batch_size
        
        # PSF parameters
        self.dx = dx
        self.dz = dz  
        self.dxpsf = dxpsf
        self.dzpsf = dzpsf
        
        # Training parameters
        self.mse_flag = mse_flag
        self.tv_weight = tv_weight
        self.hess_weight = hess_weight
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps
        
        # Load and process PSF
        self.psf = self._load_and_process_psf(psf_path)
        
        # Initialize models (matching original g, g_copy structure)
        self.model = self._create_model()  # Main training model (g)
        self.model_copy = self._create_nbr2nbr_model()  # NBR2NBR model (g_copy)
        
        # Initialize loss functions
        self.nbr2nbr_loss = create_nbr2nbr_loss(mse_flag=mse_flag)
        self.psf_loss = create_psf_loss_3d_nbr2nbr(
            psf=self.psf,
            mse_flag=mse_flag,
            upsample_flag=bool(upsample_flag),
            tv_weight=tv_weight,
            hess_weight=hess_weight,
            insert_xy=insert_xy,
            insert_z=insert_z
        )
        
        # Training step counter for learning rate scheduling
        self.training_step_count = 0
        
    def _load_and_process_psf(self, psf_path: str) -> np.ndarray:
        """Load and process PSF for training"""
        if not psf_path or not os.path.exists(psf_path):
            print(f"Warning: PSF path {psf_path} not found, creating dummy PSF")
            # Create a dummy Gaussian PSF for testing
            z, y, x = 13, 27, 27
            psf = np.zeros((z, y, x))
            psf[z//2, y//2, x//2] = 1.0
            # Apply Gaussian blur
            from scipy.ndimage import gaussian_filter
            psf = gaussian_filter(psf, sigma=[1, 2, 2])
        else:
            print(f"Loading PSF from: {psf_path}")
            psf = read_tiff_stack(psf_path)
        
        print(f"Original PSF shape: {psf.shape}")
        
        # Interpolate PSF to match training data resolution
        if abs(self.dxpsf - self.dx) > 1e-6 or abs(self.dzpsf - self.dz) > 1e-6:
            print(f"Interpolating PSF from ({self.dxpsf}, {self.dzpsf}) to ({self.dx}, {self.dz})")
            psf = interpolate_psf_3d(psf, self.dxpsf, self.dzpsf, self.dx, self.dz)
            print(f"Interpolated PSF shape: {psf.shape}")
        
        # Crop PSF for efficient computation
        sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)
        ksize = int(sigma_y * 4)
        
        z, y, x = psf.shape
        center_z, center_y, center_x = z//2, y//2, x//2
        halfz = min(z//2, max(3, self.input_z - 1))
        
        if ksize <= min(center_x, center_y):
            psf = psf[center_z-halfz:center_z+halfz+1,
                     center_y-ksize:center_y+ksize+1,
                     center_x-ksize:center_x+ksize+1]
        
        # Normalize PSF
        psf = psf / np.sum(psf)
        print(f"Final PSF shape: {psf.shape}")
        
        return psf
    
    def _create_model(self) -> nn.Module:
        """Create the main neural network model"""
        full_input_shape = (
            self.input_shape[0] + 2 * self.insert_xy,
            self.input_shape[1] + 2 * self.insert_xy, 
            self.input_shape[2] + 2 * self.insert_z,
            1
        )
        
        if self.model_name == "twostage_RCAN3D":
            model = twostage_RCAN3D.RCAN3D_factory(
                input_shape=full_input_shape,
                upsample_flag=self.upsample_flag,
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        elif self.model_name == "twostage_Unet3D":
            model = twostage_Unet3D.Unet(
                input_shape=full_input_shape,
                upsample_flag=self.upsample_flag,
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model
    
    def _create_nbr2nbr_model(self) -> nn.Module:
        """Create the NBR2NBR model (g_copy) with doubled z dimension"""
        nbr2nbr_input_shape = (
            self.input_shape[0] + 2 * self.insert_xy,
            self.input_shape[1] + 2 * self.insert_xy,
            self.input_shape[2] * 2 + 2 * self.insert_z,  # Doubled z for NBR2NBR
            1
        )
        
        if self.model_name == "twostage_RCAN3D":
            model_copy = twostage_RCAN3D.RCAN3D_factory(
                input_shape=nbr2nbr_input_shape,
                upsample_flag=0,  # No upsampling for NBR2NBR model
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        elif self.model_name == "twostage_Unet3D":
            model_copy = twostage_Unet3D.Unet(
                input_shape=nbr2nbr_input_shape,
                upsample_flag=0,  # No upsampling for NBR2NBR model
                insert_xy=self.insert_xy,
                insert_z=self.insert_z
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model_copy
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step matching original TensorFlow NBR2NBR approach"""
        input_batch, gt_batch = batch
        
        # Create NBR2NBR training data (matching original lines 446-454)
        input_g, output_g = self._create_nbr2nbr_data(input_batch, gt_batch)
        
        # Forward pass on main model
        denoise_out, deconv_out = self(input_batch)
        
        # Prepare ground truth data for loss computation
        # Stack gt with output_g for NBR2NBR loss (matching original gt setup)
        gt_combined_denoise = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        gt_combined_deconv = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        
        # Compute losses
        denoise_loss = self.nbr2nbr_loss(gt_combined_denoise, denoise_out.squeeze(1))
        deconv_loss = self.psf_loss(gt_combined_deconv, deconv_out.squeeze(1))
        
        total_loss = denoise_loss + deconv_loss
        
        # Manual learning rate scheduling (matching original)
        self.training_step_count += 1
        if self.training_step_count % 5000 == 0 or self.training_step_count % 7500 == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            new_lr = current_lr * self.lr_decay_factor
            self.trainer.optimizers[0].param_groups[0]['lr'] = new_lr
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('denoise_loss', denoise_loss, on_step=True, on_epoch=True)
        self.log('deconv_loss', deconv_loss, on_step=True, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return total_loss
    
    def _create_nbr2nbr_data(self, input_batch: torch.Tensor, gt_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create NBR2NBR training data matching original TensorFlow implementation"""
        batch_size, _, d, h, w = input_batch.shape
        input_y, input_x, input_z = self.input_shape
        
        # Create input_G with alternating GT and noisy patterns
        # Matching original: input_G = np.zeros([batch_size,input_y+insert_xy*2,input_x+insert_xy*2,(input_z+insert_z)*2,1])
        input_g = torch.zeros(
            batch_size, 1, 
            input_z * 2 + 2 * self.insert_z,
            input_y + 2 * self.insert_xy, 
            input_x + 2 * self.insert_xy,
            device=input_batch.device
        )
        
        # Fill alternating pattern (matching original loop)
        for z in range(self.insert_z, input_z * 2 + self.insert_z, 2):
            gt_z_idx = (z - self.insert_z) // 2
            input_z_idx = (z + self.insert_z) // 2
            
            if gt_z_idx < input_z:
                # Even z-indices: GT data
                input_g[:, :, z, self.insert_xy:self.insert_xy + input_y, 
                       self.insert_xy:self.insert_xy + input_x] = \
                    gt_batch[:, :, gt_z_idx, :, :]
                
                # Odd z-indices: input data
                if z + 1 < input_z * 2 + 2 * self.insert_z and input_z_idx < d:
                    input_g[:, :, z + 1, :, :] = input_batch[:, :, input_z_idx, :, :]
        
        # Copy weights from main model to NBR2NBR model (matching g_copy.set_weights(g.get_weights()))
        self.model_copy.load_state_dict(self.model.state_dict())
        
        # Forward pass through NBR2NBR model
        with torch.no_grad():
            output_g_raw, _ = self.model_copy(input_g)
            
            # Extract alternating pattern difference (matching original output_G = output_G[:,:,:,1::2,:]-output_G[:,:,:,0::2,:])
            output_g = output_g_raw[:, :, 1::2, :, :] - output_g_raw[:, :, 0::2, :, :]
        
        return input_g, output_g
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        input_batch, gt_batch = batch
        
        # Create NBR2NBR validation data
        input_g, output_g = self._create_nbr2nbr_data(input_batch, gt_batch)
        
        denoise_out, deconv_out = self(input_batch)
        
        # Prepare ground truth for validation
        gt_combined_denoise = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        gt_combined_deconv = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        
        denoise_loss = self.nbr2nbr_loss(gt_combined_denoise, denoise_out.squeeze(1))
        deconv_loss = self.psf_loss(gt_combined_deconv, deconv_out.squeeze(1))
        total_loss = denoise_loss + deconv_loss
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_denoise_loss', denoise_loss, on_step=False, on_epoch=True)
        self.log('val_deconv_loss', deconv_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers (no automatic scheduler, manual in training_step)"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                                   betas=(0.9, 0.999), weight_decay=1e-5)
        
        # No automatic scheduler - we handle learning rate manually in training_step
        # to match the original TensorFlow implementation exactly
        return optimizer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ZS-DeconvNet 3D')
    
    # Model parameters
    parser.add_argument("--model", type=str, default="twostage_RCAN3D", 
                       choices=["twostage_RCAN3D", "twostage_Unet3D"])
    parser.add_argument("--upsample_flag", type=int, default=0)
    
    # Training settings
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--valid_interval", type=int, default=1000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--precision", default='16-mixed')
    
    # Learning rate
    parser.add_argument("--start_lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    
    # Data paths
    parser.add_argument("--save_weights_dir", type=str, default="./outputs")
    parser.add_argument("--save_weights_suffix", type=str, default="_Hess0.1_MAE_up")
    parser.add_argument("--psf_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_images_path", type=str)
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--background", type=int, default=100)
    
    # Input parameters
    parser.add_argument("--input_y", type=int, default=64)
    parser.add_argument("--input_x", type=int, default=64)
    parser.add_argument("--input_z", type=int, default=13)
    parser.add_argument("--insert_z", type=int, default=2)
    parser.add_argument("--insert_xy", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--dx", type=float, default=0.0926)
    parser.add_argument("--dz", type=float, default=0.3704)
    parser.add_argument("--dxpsf", type=float, default=0.0926)
    parser.add_argument("--dzpsf", type=float, default=0.05)
    parser.add_argument("--norm_flag", type=int, default=0)
    
    # Loss function parameters
    parser.add_argument("--mse_flag", type=int, default=0)
    parser.add_argument("--TV_weight", type=float, default=0)
    parser.add_argument("--Hess_weight", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Set up paths
    save_path = os.path.join(args.save_weights_dir, 
                            f"{args.model}_{args.folder}{args.save_weights_suffix}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create simplified data module (matching original TensorFlow approach)
    data_module = SimplifiedDataModule(
        data_dir=args.data_dir,
        folder=args.folder,
        input_shape=(args.input_y, args.input_x, args.input_z),
        insert_xy=args.insert_xy,
        insert_z=args.insert_z,
        batch_size=args.batch_size,
        norm_flag=args.norm_flag,
        load_all_data=True
    )
    
    # Create model manually (matching original TensorFlow g model creation)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    
    model = ZSDeconvNet3DModule(
        model_name=args.model,
        psf_path=args.psf_path,
        input_shape=(args.input_y, args.input_x, args.input_z),
        upsample_flag=args.upsample_flag,
        insert_xy=args.insert_xy,
        insert_z=args.insert_z,
        dx=args.dx,
        dz=args.dz,
        dxpsf=args.dxpsf,
        dzpsf=args.dzpsf,
        mse_flag=args.mse_flag,
        tv_weight=args.TV_weight,
        hess_weight=args.Hess_weight,
        learning_rate=args.start_lr,
        lr_decay_factor=args.lr_decay_factor,
        batch_size=args.batch_size
    ).to(device)
    
    # Setup optimizer (matching original Adam setup)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, 
                               betas=(0.9, 0.999), weight_decay=1e-5)
    
    # Setup logging directory
    log_path = os.path.join(save_path, 'graph')
    if os.path.exists(log_path):
        for file_name in os.listdir(log_path):
            path_file = os.path.join(log_path, file_name)
            if os.path.isfile(path_file):
                os.remove(path_file)
    else:
        os.makedirs(log_path)
    
    # TensorBoard writer for logging
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_path)
    
    # Manual training loop (matching original TensorFlow implementation)
    print(f"Starting training with model: {args.model}")
    print(f"Data directory: {data_module.train_images_path}")
    print(f"PSF path: {args.psf_path}")
    print(f"Output directory: {save_path}")
    
    # Training loop variables
    loss_denoise = []
    loss_deconv = []
    start_time = datetime.datetime.now()
    
    model.train()
    
    # Main training loop (matching original for loop)
    for it in range(args.iterations):
        # Get training batch
        input_batch, gt_batch = data_module.get_batch()
        input_batch = input_batch.to(device)
        gt_batch = gt_batch.to(device)
        
        # Create NBR2NBR data (matching original lines 446-454)
        input_g, output_g = model._create_nbr2nbr_data(input_batch, gt_batch)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        denoise_out, deconv_out = model(input_batch)
        
        # Prepare ground truth for loss computation
        gt_combined_denoise = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        gt_combined_deconv = torch.stack([gt_batch.squeeze(1), output_g], dim=-1)
        
        # Compute losses
        denoise_loss = model.nbr2nbr_loss(gt_combined_denoise, denoise_out.squeeze(1))
        deconv_loss = model.psf_loss(gt_combined_deconv, deconv_out.squeeze(1))
        
        total_loss = denoise_loss + deconv_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        loss_denoise.append(denoise_loss.item())
        loss_deconv.append(deconv_loss.item())
        
        # Learning rate decay (matching original manual scheduling)
        if (it + 1) % 5000 == 0 or (it + 1) % 7500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr * args.lr_decay_factor
            optimizer.param_groups[0]['lr'] = new_lr
            print(f"Learning rate decayed to: {new_lr}")
        
        # Print progress
        elapsed_time = datetime.datetime.now() - start_time
        print(f"{it + 1} it: time: {elapsed_time}, denoise_loss = {denoise_loss.item():.3e}, deconv_loss = {deconv_loss.item():.3e}")
        
        # Validation and logging (matching original intervals)
        if (it + 1) % args.valid_interval == 0 or it == 0:
            # Validation step would go here (simplified for now)
            pass
        
        if (it + 1) % args.test_interval == 0 or it == 0:
            # Write logs to TensorBoard
            writer.add_scalar('NBR2NBR_loss', np.mean(loss_denoise), it + 1)
            writer.add_scalar('deconv_loss', np.mean(loss_deconv), it + 1)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], it + 1)
            
            # Clear loss history
            loss_denoise = []
            loss_deconv = []
            
            # Save model weights (matching original save format)
            model_save_path = os.path.join(save_path, f'model_weights_{it + 1}.pth')
            torch.save({
                'iteration': it + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'denoise_loss': denoise_loss,
                'deconv_loss': deconv_loss,
            }, model_save_path)
            print(f"Model saved: {model_save_path}")
    
    writer.close()
    print("Training completed!")
    print(f"Models saved in: {save_path}")


if __name__ == "__main__":
    main()