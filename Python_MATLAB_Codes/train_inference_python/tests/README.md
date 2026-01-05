# Testing Suite for ZS-DeconvNet

This directory contains comprehensive tests for the patch-based super-resolution processing functionality in ZS-DeconvNet.

## Overview

The test suite validates the core functionality of:
- **Patch Segmentation**: Breaking large images/volumes into smaller patches
- **Super-Resolution Processing**: Processing patches with optional 2x upsampling
- **Patch Fusion**: Reassembling processed patches back into complete images/volumes
- **TIFF I/O**: Reading and writing TIFF files correctly

## Test Files

### Unit Tests
- `test_patch_processing_2d.py`: Tests for 2D patch segmentation and fusion logic
  - Segmentation with and without overlap
  - Patch extraction validation
  - Fusion with and without super-resolution
  
- `test_patch_processing_3d.py`: Tests for 3D patch segmentation and fusion logic
  - 3D volume segmentation
  - 3D patch extraction
  - 3D fusion with super-resolution in XY plane

### Integration Tests
- `test_integration.py`: End-to-end tests with actual TIFF file I/O
  - Loading and saving 2D/3D TIFF files
  - Complete workflow testing
  - Super-resolution output validation

## Running the Tests

### Run all tests:
```bash
cd /path/to/Python_MATLAB_Codes/train_inference_python
python -m pytest tests/ -v
```

Or using unittest:
```bash
cd /path/to/Python_MATLAB_Codes/train_inference_python
python -m unittest discover tests -v
```

### Run specific test file:
```bash
python -m unittest tests.test_patch_processing_2d -v
python -m unittest tests.test_patch_processing_3d -v
python -m unittest tests.test_integration -v
```

### Run specific test class or method:
```bash
python -m unittest tests.test_patch_processing_2d.TestPatchSegmentation2D -v
python -m unittest tests.test_patch_processing_2d.TestPatchSegmentation2D.test_basic_segmentation_no_overlap -v
```

## What the Tests Validate

### Patch Segmentation
- Correct calculation of patch dimensions based on image size and number of windows
- Proper handling of overlapping regions
- Coverage of entire image/volume with patches
- Edge case handling (non-divisible dimensions)

### Patch Fusion
- Accurate reconstruction of original image from patches (identity test)
- Proper handling of overlapping regions during fusion
- Correct super-resolution output dimensions (2x upsampling)
- No gaps or missing regions in fused output

### Integration
- TIFF file reading and writing preserves data
- Percentile normalization works correctly
- Complete workflow from TIFF input to TIFF output
- Output dimensions match expected values

## Test Coverage

The tests cover the key algorithms used in:
- `Infer_2D.py`: 2D inference with patch-based processing
- `Infer_3D.py`: 3D inference with patch-based processing
- `utils/utils.py`: Utility functions like `prctile_norm`

## Dependencies

The tests require:
- numpy
- tifffile
- unittest (Python standard library)

Optional:
- pytest (for enhanced test running and reporting)

## Adding New Tests

When adding new functionality:
1. Add unit tests to validate the core logic
2. Add integration tests to validate end-to-end workflows
3. Follow the existing test structure and naming conventions
4. Ensure tests are independent and can run in any order

## Continuous Integration

These tests can be integrated into CI/CD pipelines to automatically validate:
- Pull requests before merging
- Releases before deployment
- Nightly builds for regression detection
