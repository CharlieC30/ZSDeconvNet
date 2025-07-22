# PyTorch Pure Deconvolution Network

PyTorch Lightning implementation of ZS-DeconvNet for pure deconvolution with 2x super-resolution. Uses self-supervised learning with PSF-based physics loss.

## Installation

```bash
conda create -n zs-deconvnet_pytorch python=3.9
conda activate zs-deconvnet_pytorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data Structure

```
Data/
├── Train/              # Training images
├── InferenceInput/     # Input images
├── InferenceResult/    # Output (auto-created)
├── Output/             # Training outputs
└── PSF/               # PSF files
```

## Training

```bash
python train.py \
    --config config.yaml \
    --data_dir Data \
    --psf_path Data/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif \
    --output_dir Data/Output \
    --gpus 1
```

## Inference

### Standard Inference
```bash
python infer.py \
    --checkpoint Data/Output/final_model.ckpt \
    --input_dir Data/InferenceInput \
    --output_dir Data/InferenceResult \
    --device auto
```

### Large Images (Tiling)
```bash
python infer.py \
    --checkpoint Data/Output/final_model.ckpt \
    --input_dir Data/InferenceInput \
    --output_dir Data/InferenceResult \
    --tile_size 512 \
    --overlap 64 \
    --device auto
```

### Original Tiling Method
Best quality for large images:
```bash
python infer.py \
    --checkpoint Data/Output/final_model.ckpt \
    --input_dir Data/InferenceInput \
    --output_dir Data/InferenceResult \
    --use_original_tiling \
    --num_seg_window_x 4 \
    --num_seg_window_y 4 \
    --overlap_x 20 \
    --overlap_y 20 \
    --device auto
```

## Key Parameters

### Training
- `batch_size: 4` - Training batch size
- `max_epochs: 100` - Training epochs  
- `lr: 5e-5` - Learning rate
- `hessian_weight: 0.02` - Edge preservation

### Inference
- `--tile_size 512` - Tile size for large images
- `--overlap 64` - Overlap between tiles
- `--use_original_tiling` - Use original tiling algorithm
- `--num_seg_window_x/y 4` - Number of segments per dimension
- `--overlap_x/y 20` - Overlap for original tiling

## Troubleshooting

1. **CUDA out of memory**: Reduce `batch_size` or use tiling for inference
2. **Model loading warnings**: Safe to ignore
3. **Black outputs**: Check `hessian_weight: 0.02` in config
4. **Large images**: Use `--use_original_tiling`

## Output

- Training: `Data/Output/deconv_logs/MMDD_HHMM/`
- Final model: `Data/Output/final_model.ckpt`
- Results: `*_deconvolved.tif` (16-bit TIFF, 2x resolution)