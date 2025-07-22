import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import argparse
from pathlib import Path


def load_psf(psf_path, device='cpu'):
    """Load and normalize PSF from file."""
    psf = np.float32(tifffile.imread(psf_path))
    if len(psf.shape) == 3:
        psf = psf[:, :, 0]
    psf = psf / np.sum(psf)
    return torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)


def load_image(image_path, mode='slice'):
    """Load image with different processing modes."""
    img = np.array(tifffile.imread(image_path)).astype(np.float32)
    
    if len(img.shape) == 3:
        if mode == 'sum':
            img = np.sum(img, axis=0)
            img = img / 65535.0
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            return img_tensor, False
        elif mode == 'slice':
            return img, True
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    img = img / 65535.0
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return img_tensor, False


def apply_psf_blur(image_tensor, psf_tensor):
    """Apply PSF blur using 2D convolution."""
    return F.conv2d(image_tensor, psf_tensor, padding='same')


def process_3d_slices(img_3d, psf_tensor, device='cpu'):
    """Process 3D image stack slice by slice."""
    num_slices = img_3d.shape[0]
    processed_slices = []
    
    print(f"Processing {num_slices} slices...")
    
    for slice_idx in range(num_slices):
        slice_2d = img_3d[slice_idx] / 65535.0
        slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0).to(device)
        blurred_slice = apply_psf_blur(slice_tensor, psf_tensor)
        processed_slice = blurred_slice.squeeze().cpu().numpy()
        processed_slices.append(processed_slice)
        
        if (slice_idx + 1) % 30 == 0 or slice_idx == num_slices - 1:
            print(f"  Slice {slice_idx + 1}/{num_slices}")
    
    return np.stack(processed_slices, axis=0)


def save_image(data, output_path):
    """Save processed image as 16-bit TIFF."""
    if isinstance(data, torch.Tensor):
        img_np = data.squeeze().cpu().numpy()
    else:
        img_np = data
    
    img_np = np.clip(img_np, 0, 1)
    img_uint16 = (img_np * 65535).astype(np.uint16)
    tifffile.imwrite(str(output_path), img_uint16)


def main():
    parser = argparse.ArgumentParser(description='PSF Blur Test')
    parser.add_argument('--image', type=str, required=True, help='Input image filename (in PSFtest_input/)')
    parser.add_argument('--psf', type=str, required=True, help='PSF filename (in PSFoutput/)')
    parser.add_argument('--mode', type=str, default='slice', choices=['slice', 'sum'], 
                       help='How to handle 3D images (default: slice)')
    
    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print("Using GPU for processing")
    else:
        device = 'cpu'
        print("GPU not available - using CPU")
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "PSFtest" / "PSFtest_input"
    output_dir = base_dir / "PSFtest" / "PSFtest_output"
    psf_dir = base_dir / "PSFoutput"
    
    # Input image path
    image_path = input_dir / args.image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    # Find PSF file
    psf_path = None
    for subdir in ['optical', 'gaussian']:
        potential_path = psf_dir / subdir / args.psf
        if potential_path.exists():
            psf_path = potential_path
            break
    
    if psf_path is None:
        potential_path = psf_dir / args.psf
        if potential_path.exists():
            psf_path = potential_path
    
    if psf_path is None:
        print(f"PSF not found: {args.psf}")
        print(f"Searched in: {psf_dir}/optical/, {psf_dir}/gaussian/, {psf_dir}/")
        return
    
    image_data, is_3d = load_image(image_path, args.mode)
    psf_tensor = load_psf(psf_path, device)
    
    # Apply blur processing
    if is_3d and args.mode == 'slice':
        blurred_3d = process_3d_slices(image_data, psf_tensor, device)
        image_stem = image_path.stem
        psf_stem = psf_path.stem
        blurred_output = output_dir / f"blurred_{image_stem}_{psf_stem}_3D.tif"
        save_image(blurred_3d, blurred_output)
        print(f"Saved: {blurred_output}")
        
    else:
        image_tensor = image_data.to(device)
        blurred_tensor = apply_psf_blur(image_tensor, psf_tensor)
        image_stem = image_path.stem
        psf_stem = psf_path.stem
        mode_suffix = "_2D" if args.mode == 'sum' else ""
        blurred_output = output_dir / f"blurred_{image_stem}_{psf_stem}{mode_suffix}.tif"
        save_image(blurred_tensor, blurred_output)
        print(f"Saved: {blurred_output}")


if __name__ == "__main__":
    main()

# Example usage:
# python test_psf.py --image xyft0.tif --psf gaussian/PSF_gaussian_sigma4.0_size257.tif --mode slice
# python test_psf.py --image xyft0.tif --psf optical/PSF_optical_NA1.1_lambda525_size257.tif --mode slice