import numpy as np
import tifffile
import os
from pathlib import Path

def generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI):
    """Generate optical PSF (equivalent to MATLAB XxPSFGenerator.m)"""
    
    # Frequency space setup
    dk = 2 * np.pi / dxy / SizeXY
    kx = np.arange(-(SizeXY-1)//2, (SizeXY-1)//2 + 1) * dk
    kx, ky = np.meshgrid(kx, kx)
    kr_sq = kx**2 + ky**2
    z = np.arange(-(SizeZ-1)//2, (SizeZ-1)//2 + 1) * dz
    
    # Optical system
    PupilMask = (kr_sq <= (2*np.pi/wavelength*NA)**2)
    kz_temp = (2*np.pi/wavelength*RI)**2 - kr_sq
    kz_temp = np.where(kz_temp >= 0, kz_temp, 0)
    kz = np.sqrt(kz_temp) * PupilMask
    
    # Calculate PSF for each layer
    PSF = np.zeros((SizeXY, SizeXY, SizeZ), dtype=np.float64)
    
    for ii in range(SizeZ):
        phase_term = PupilMask * np.exp(1j * kz * z[ii])
        fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase_term)))
        PSF[:, :, ii] = np.abs(fft_result)**2
    
    return PSF

def generate_gaussian_psf(size, sigma):
    """Generate Gaussian PSF (equivalent to MATLAB XxGuassianGenerator2D.m)"""
    
    # Match MATLAB's fspecial('gaussian') implementation
    # Create coordinate arrays
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)
    
    # Generate Gaussian kernel (matches MATLAB fspecial)
    gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Normalize (matches MATLAB normalization)
    gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
    
    # Convert to 3D format for consistency
    gaussian_3d = gaussian_2d[:, :, np.newaxis]
    
    return gaussian_3d

def save_psf_tiff(psf, output_path):
    """Save PSF as TIFF format"""
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize to 16-bit range
    psf_normalized = psf / np.max(psf) * (2**15)
    psf_uint16 = psf_normalized.astype(np.uint16)
    
    # Save TIFF
    if psf.shape[2] == 1:
        tifffile.imwrite(str(output_path), psf_uint16[:, :, 0])
    else:
        tifffile.imwrite(str(output_path), psf_uint16)
    
    return str(output_path)

def create_optical_psf_file(dxy=92.6, dz=92.6, SizeXY=257, SizeZ=1, wavelength=525, NA=1.1, RI=1.3):
    """Generate optical PSF and save with descriptive filename"""
    
    print("Generating optical PSF...")
    print(f"Parameters: {SizeXY}x{SizeXY}x{SizeZ}, dxy={dxy}nm, wavelength={wavelength}nm, NA={NA}")
    
    # Generate PSF
    psf = generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI)
    
    # Create descriptive filename
    filename = f"PSF_optical_NA{NA}_lambda{wavelength}_size{SizeXY}"
    if SizeZ > 1:
        filename += f"x{SizeZ}"
    filename += ".tif"
    
    # Save to optical subdirectory
    output_path = Path("/home/aero/charliechang/projects/ZS-DeconvNet/Python_PSF/output/optical") / filename
    saved_path = save_psf_tiff(psf, output_path)
    
    print(f"Optical PSF saved: {saved_path}")
    print(f"Shape: {psf.shape}, Max: {np.max(psf):.2e}")
    print(f"Training parameters: --otf_or_psf_path '{saved_path}' --psf_src_mode 1 --dxypsf {dxy/1000}")
    
    return saved_path

def create_gaussian_psf_file(size=31, sigma=3.0):
    """Generate Gaussian PSF and save with descriptive filename"""
    
    print("Generating Gaussian PSF...")
    print(f"Parameters: {size}x{size}, sigma={sigma}")
    
    # Generate PSF
    psf = generate_gaussian_psf(size, sigma)
    
    # Create descriptive filename
    filename = f"PSF_gaussian_sigma{sigma}_size{size}.tif"
    
    # Save to gaussian subdirectory
    output_path = Path("/home/aero/charliechang/projects/ZS-DeconvNet/Python_PSF/output/gaussian") / filename
    saved_path = save_psf_tiff(psf, output_path)
    
    print(f"Gaussian PSF saved: {saved_path}")
    print(f"Shape: {psf.shape}, Max: {np.max(psf):.2e}")
    
    return saved_path

def create_psf():
    """Legacy function for backward compatibility"""
    print("Optical PSF Generator")
    return create_optical_psf_file()

if __name__ == "__main__":
    # create_optical_psf_file()
    # create_gaussian_psf_file()
    
    create_optical_psf_file(dxy=31.3, dz=31.3, SizeXY=257, SizeZ=1, wavelength=525, NA=1.3, RI=1.3)
    # create_gaussian_psf_file(size=257, sigma=4.0)