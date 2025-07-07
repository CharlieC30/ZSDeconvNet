# CLAUDE.md

## Project Overview

ZS-DeconvNet implements self-supervised deep learning for simultaneous denoising and super-resolution in fluorescence microscopy without requiring paired training data.

## Architecture

Four main implementations:
1. **Python/TensorFlow**: Primary implementation in `Python_MATLAB_Codes/train_inference_python/`
2. **MATLAB**: Data preprocessing and PSF generation
3. **Fiji Plugin**: ImageJ/Fiji end-user plugin
4. **PyTorch**: Pure deconvolution implementation in `PyTorch_Deconv/`

## Environment Setup

```bash
conda activate zs-deconvnet_pytorch
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt
conda install cudatoolkit==11.3.1 cudnn==8.2.1
```

## Conda Virtual Environments

- Using conda virtual environment `zs-deconvnet_pytorch` for project development and experimentation

[Rest of the existing content remains unchanged]