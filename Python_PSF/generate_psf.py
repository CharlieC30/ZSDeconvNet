import numpy as np
import tifffile
from pathlib import Path

def generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI):
    """
    Generate optical PSF based on Born & Wolf scalar diffraction theory.
    
    Parameters
    ----------
    dxy, dz : float
        Pixel size in nanometers (nm)
    SizeXY, SizeZ : int
        PSF dimensions in pixels
    wavelength : float
        Emission wavelength in nm
    NA : float
        Numerical aperture
    RI : float
        Refractive index of medium
    
    Returns
    -------
    numpy.ndarray
        3D PSF array with shape (SizeXY, SizeXY, SizeZ) - (Y, X, Z) order
    """
    
    # Frequency space coordinates
    dk = 2 * np.pi / dxy / SizeXY
    kx = np.arange(-(SizeXY-1)//2, (SizeXY-1)//2 + 1) * dk
    kx, ky = np.meshgrid(kx, kx)
    kr_sq = kx**2 + ky**2
    z = np.arange(-(SizeZ-1)//2, (SizeZ-1)//2 + 1) * dz
    
    # Pupil function and wave vector components
    PupilMask = (kr_sq <= (2*np.pi/wavelength*NA)**2)
    kz_temp = (2*np.pi/wavelength*RI)**2 - kr_sq
    kz_temp = np.where(kz_temp >= 0, kz_temp, 0)
    kz = np.sqrt(kz_temp) * PupilMask
    
    # Generate PSF for each z-layer
    PSF = np.zeros((SizeXY, SizeXY, SizeZ), dtype=np.float64)
    
    for ii in range(SizeZ):
        phase_term = PupilMask * np.exp(1j * kz * z[ii])
        fft_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase_term)))
        PSF[:, :, ii] = np.abs(fft_result)**2
    
    return PSF

def generate_gaussian_psf(size, sigma):
    """
    Generate normalized Gaussian PSF for simplified blur simulation.
    
    Parameters
    ----------
    size : int
        Kernel size in pixels
    sigma : float
        Standard deviation in pixels
    
    Returns
    -------
    numpy.ndarray
        3D array with shape (size, size, 1)
    """
    
    # Create coordinate grids
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    X, Y = np.meshgrid(x, y)
    
    # Generate normalized Gaussian
    gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
    
    # Convert to 3D format
    gaussian_3d = gaussian_2d[:, :, np.newaxis]
    
    return gaussian_3d

def save_psf_tiff(psf, output_path):
    """
    Save PSF as 16-bit TIFF file.
    
    Parameters
    ----------
    psf : numpy.ndarray
        PSF array to save
    output_path : str or Path
        Output file path
    
    Returns
    -------
    str
        Path to the saved file
    """
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize to 16-bit range
    psf_normalized = psf / np.max(psf) * (2**15)
    psf_uint16 = psf_normalized.astype(np.uint16)
    
    # Save as TIFF with correct axis order for compatibility
    if psf.shape[2] == 1:
        # Single slice: save as 2D TIFF
        tifffile.imwrite(str(output_path), psf_uint16[:, :, 0])
    else:
        # Multi-slice: transpose to (Z,Y,X) for correct TIFF page order
        # This ensures compatibility with imageio.mimread() in TensorFlow training
        psf_transposed = np.transpose(psf_uint16, (2, 0, 1))  # (Y,X,Z) -> (Z,Y,X)
        tifffile.imwrite(str(output_path), psf_transposed)
    
    return str(output_path)

def create_optical_psf_file(dxy=92.6, dz=92.6, SizeXY=257, SizeZ=1, wavelength=525, NA=1.1, RI=1.3):
    """Generate and save optical PSF with auto-generated filename."""
    
    psf = generate_optical_psf(dxy, dz, SizeXY, SizeZ, wavelength, NA, RI)
    
    filename = f"PSF_optical_NA{NA}_lambda{wavelength}_size{SizeXY}"
    if SizeZ > 1:
        filename += f"_Z{SizeZ}"
    filename += ".tif"
    
    base_dir = Path(__file__).parent
    output_path = base_dir / "PSFoutput" / "optical" / filename
    saved_path = save_psf_tiff(psf, output_path)
    
    print(f"Generated optical PSF: {filename}")
    
    return saved_path

def create_gaussian_psf_file(size=31, sigma=3.0):
    """Generate and save Gaussian PSF with auto-generated filename."""
    
    psf = generate_gaussian_psf(size, sigma)
    
    filename = f"PSF_gaussian_sigma{sigma}_size{size}.tif"
    
    base_dir = Path(__file__).parent
    output_path = base_dir / "PSFoutput" / "gaussian" / filename
    saved_path = save_psf_tiff(psf, output_path)
    
    print(f"Generated Gaussian PSF: {filename} (σ={sigma} pixels)")
    
    return saved_path

if __name__ == "__main__":
    create_optical_psf_file(dxy=31.3, dz=31.3, SizeXY=79, SizeZ=23, wavelength=525, NA=0.8, RI=1.3)
    # create_gaussian_psf_file(size=257, sigma=4.0)