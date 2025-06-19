#!/usr/bin/env python3
"""
Basic test script to check if dependencies are working.
"""

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning version: {pl.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Lightning not available: {e}")

try:
    import yaml
    print(f"✓ PyYAML is available")
except ImportError as e:
    print(f"✗ PyYAML not available: {e}")

try:
    import tifffile
    print(f"✓ tifffile is available")
except ImportError as e:
    print(f"✗ tifffile not available: {e}")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy not available: {e}")

try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV not available: {e}")

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy not available: {e}")

print("\nTesting basic functionality...")

try:
    # Test basic tensor operations
    x = torch.randn(2, 3)
    print(f"✓ Basic PyTorch tensor operations work")
except Exception as e:
    print(f"✗ PyTorch tensor operations failed: {e}")

try:
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available with {torch.cuda.device_count()} device(s)")
        print(f"  Current device: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA is not available, will use CPU")
except Exception as e:
    print(f"⚠ CUDA check failed: {e}")

print("\nDependency check complete!")