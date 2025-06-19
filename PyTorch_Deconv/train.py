#!/usr/bin/env python3
"""
Training script for PyTorch Lightning deconvolution model.
"""

import os
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# Add src to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.models.lightning_module import create_lightning_module
from src.data.datamodule import DeconvDataModule


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_callbacks(config):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        filename='{epoch}-{val_loss:.4f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_config = config.get('early_stopping', {})
    if early_stop_config.get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_loss'),
            mode=early_stop_config.get('mode', 'min'),
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001)
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    parser = argparse.ArgumentParser(description='Train deconvolution model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--psf_path', type=str, required=True,
                       help='Path to PSF file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a fast development run for debugging')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update paths in config
    config['psf_path'] = args.psf_path
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    data_config = config.get('data', {})
    datamodule = DeconvDataModule(
        data_dir=args.data_dir,
        batch_size=data_config.get('batch_size', 4),
        patch_size=data_config.get('patch_size', 128),
        insert_xy=data_config.get('insert_xy', 16),
        num_workers=data_config.get('num_workers', 4),
        train_val_split=data_config.get('train_val_split', 0.8)
    )
    
    # Create model
    model = create_lightning_module(config)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='deconv_logs',
        version=None
    )
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Setup trainer
    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        accelerator='gpu' if args.gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.gpus if args.gpus > 0 and torch.cuda.is_available() else 'auto',
        logger=logger,
        callbacks=callbacks,
        precision=trainer_config.get('precision', 32),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 0.5),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Print configuration
    print("=" * 50)
    print("Training Configuration:")
    print(f"Data directory: {args.data_dir}")
    print(f"PSF path: {args.psf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max epochs: {trainer_config.get('max_epochs', 100)}")
    print(f"Batch size: {data_config.get('batch_size', 4)}")
    print(f"Learning rate: {config['optimizer']['lr']}")
    print(f"Device: {'GPU' if args.gpus > 0 and torch.cuda.is_available() else 'CPU'}")
    print("=" * 50)
    
    # Train model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.fit(model, datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule)
    
    # Save final model
    final_model_path = output_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Test the model
    if not args.fast_dev_run:
        print("Running final validation...")
        trainer.validate(model, datamodule)


if __name__ == '__main__':
    main()