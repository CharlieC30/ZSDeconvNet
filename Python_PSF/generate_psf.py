import numpy as np
import tifffile
import os
import math

def generate_psf(dxy, dz, size_xy, size_z, lamd, na, ri, file_name):
    """
    Generates a theoretical 3D wide-field microscope Point Spread Function (PSF)
    based on optical parameters.

    Args:
        dxy (float): Lateral sampling interval in the image plane (nm).
        dz (float): Axial sampling interval in the image plane (nm).
        size_xy (int): Lateral size (number of pixels) of the PSF in x and y.
                       Should be an odd number for the center to be a pixel.
        size_z (int): Axial size (number of slices) of the PSF in z.
                      Should be an odd number for the center to be a slice.
        lamd (float): Emission wavelength (nm).
        na (float): Numerical Aperture of the objective.
        ri (float): Refractive Index of the immersion medium.
        file_name (str): Full path including filename to save the generated PSF (.tif).
    """
    if size_xy % 2 == 0 or size_z % 2 == 0:
        print("Warning: SizeXY and SizeZ should ideally be odd numbers for the center pixel/slice.")

    # Calculate frequency domain parameters
    # Spatial frequencies kx, ky
    dk = 2 * np.pi / (dxy * size_xy)
    kx = np.arange(-(size_xy - 1) / 2, (size_xy - 1) / 2 + 1) * dk
    ky = np.arange(-(size_xy - 1) / 2, (size_xy - 1) / 2 + 1) * dk
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy') # Match MATLAB's meshgrid behavior
    kr_sq = kx_grid**2 + ky_grid**2

    # Axial spatial coordinates z
    z = np.arange(-(size_z - 1) / 2, (size_z - 1) / 2 + 1) * dz

    # Define the Pupil Mask in frequency space
    # The cutoff frequency is 2*pi*NA/lambda
    pupil_mask = (kr_sq <= (2 * np.pi * na / lamd)**2)

    # Calculate axial frequency component kz
    # kz = sqrt((2*pi*RI/lambda)^2 - kr_sq)
    kz_sq = (2 * np.pi * ri / lamd)**2 - kr_sq
    # Ensure kz is real where pupil_mask is True, handle potential small negative values due to precision
    kz = np.sqrt(np.maximum(0, kz_sq)) * pupil_mask

    # Initialize the 3D PSF array
    psf_3d = np.zeros((size_xy, size_xy, size_z), dtype=np.float64)

    # Calculate PSF slice by slice for each axial position
    for ii in range(size_z):
        # Calculate the complex amplitude in frequency space for this z slice
        # This is the Pupil Function multiplied by a phase term exp(i * kz * z)
        tmp_freq = pupil_mask * np.exp(1j * kz * z[ii])

        # Perform 2D inverse Fourier Transform to get the Amplitude Spread Function (ASF)
        # Use ifftshift before ifft2 and fftshift after ifft2 to handle frequency centering
        tmp_spatial = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(tmp_freq)))

        # The Intensity PSF is the square of the magnitude of the ASF
        psf_3d[:, :, ii] = np.abs(tmp_spatial)**2

    # Normalize the PSF and convert to uint16
    # Normalize to max 1, then scale to 2^15 for uint16 storage
    psf_3d = psf_3d / np.max(psf_3d) * (2**15)
    psf_uint16 = psf_3d.astype(np.uint16)

    # Ensure the output directory exists
    output_dir = os.path.dirname(file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the 3D PSF as a multi-page TIFF file
    # tifffile saves in ZYX order by default for 3D arrays
    tifffile.imwrite(file_name, psf_uint16, photometric='minisblack')

    print(f"Generated PSF saved to: {file_name}")

# --- Main execution block ---
if __name__ == "__main__":
    # Define parameters (similar to the MATLAB script)
    dxy = 92.6e-3  # lateral sampling, in um (converted from nm)
    dz = 92.6e-3   # axial sampling, in um (converted from nm)
    size_xy = 27   # lateral pixel number of PSF
    size_z = 13    # axial pixel number of PSF
    lamd = 525e-3  # emission wavelength, in um (converted from nm)
    na = 1.1       # numerical aperture
    ri = 1.3       # refractive index

    # Define the output file path
    # Assumes the script is run from within the ZS-DeconvNet project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'generated_psfs')
    file_name = os.path.join(output_dir, 'SimulatedPSF.tif')

    # Generate and save the PSF
    generate_psf(dxy, dz, size_xy, size_z, lamd, na, ri, file_name)