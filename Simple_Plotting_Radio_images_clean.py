#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple_Plotting_Radio_images_clean.py
Cleaned and refactored version of user's original Simple_Plotting_Radio_images.py

Main changes & aims:
- Group imports and remove duplicates.
- Add docstrings and comments explaining each function and plotting block.
- Make helper functions reusable (loadfits, define_contours, crop, plotting helpers).
- Use safe defaults and catch missing files gracefully.
- Keep plotting behavior identical where possible but clearer and modular.
"""

from typing import Tuple, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import matplotlib.colors as colors

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import median_absolute_deviation as apy_mad
from astropy.visualization import simple_norm, ImageNormalize, ZScaleInterval
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.ndimage import gaussian_filter
from reproject import reproject_interp
from regions import Regions

# ---------------------------
# Helper functions
# ---------------------------

def loadfits(fits_file: str, index: int = 0) -> Tuple[np.ndarray, WCS]:
    """Load FITS image, select appropriate slice if NAXIS>2, trim all-NaN borders, and return (image, wcs).

    Parameters
    ----------
    fits_file: path to FITS file
    index: HDU index to open (default 0)

    Returns
    -------
    (image_2d, wcs2d)
    """
    with fits.open(fits_file) as hdul:
        hdu = hdul[index]
        header = hdu.header
        data = hdu.data

        if data is None:
            raise ValueError(f"No data found in {fits_file} HDU {index}")

        naxis = header.get('NAXIS', data.ndim)
        if naxis >= 4 and data.ndim >= 4:
            img = data[0, 0, :, :]
        elif naxis == 3 and data.ndim == 3:
            img = data[0, :, :]
        else:
            img = data

        img = np.squeeze(img)
        if img.ndim != 2:
            raise ValueError(f"Unexpected image dimensions after squeezing: {img.shape}")

        # trim all-NaN rows/cols
        valid_rows = np.any(~np.isnan(img), axis=1)
        valid_cols = np.any(~np.isnan(img), axis=0)
        img_trimmed = img[valid_rows][:, valid_cols]

        wcs2d = WCS(header, naxis=2)

    return img_trimmed, wcs2d


def define_contours(imdata: np.ndarray, nlevels: int = 3, maxval: float = 0.8) -> np.ndarray:
    """Estimate contour levels from image MAD (3σ, 5σ, plus a few linear steps).

    This mirrors the original behavior used for radio/X-ray contour generation.
    """
    finite = imdata[np.isfinite(imdata)]
    if finite.size == 0:
        raise ValueError("Input contains no finite values for MAD-based contours")
    im_mad = apy_mad(finite)
    mylevels = np.array([3.0, 5.0]) * im_mad
    dat_max = np.nanmax(imdata) * maxval
    if dat_max > im_mad * 5:
        step = (dat_max - mylevels[-1]) / nlevels
        for _ in range(nlevels):
            mylevels = np.append(mylevels, mylevels[-1] + step)
    return mylevels


def define_contours_from_noise(im_max: float, noise: float, minsig: int = 5, max_n: int = 10, maxval: float = 0.8) -> np.ndarray:
    """Define contour levels from a known noise RMS and a peak value.

    Useful when you know the map RMS rather than estimating from data.
    """
    minlevel = noise * minsig
    dat_max = im_max * maxval
    levels = np.array([minlevel])
    while levels[-1] < dat_max:
        levels = np.append(levels, levels[-1] * np.sqrt(2.0))
        if len(levels) > max_n:
            break
    return levels


def crop_image(ra_min: float, ra_max: float, dec_min: float, dec_max: float, img_data: np.ndarray, wcs_coord: WCS) -> Tuple[np.ndarray, WCS]:
    """Crop an image array to the bounding box given by RA/Dec (degrees)."""
    a_min, b_min = wcs_coord.world_to_pixel(SkyCoord(ra=ra_min, dec=dec_min, unit='deg'))
    a_max, b_max = wcs_coord.world_to_pixel(SkyCoord(ra=ra_max, dec=dec_max, unit='deg'))
    x_min, x_max = int(min(a_min, a_max)), int(max(a_min, a_max))
    y_min, y_max = int(min(b_min, b_max)), int(max(b_min, b_max))
    # clip to bounds
    y_min = max(0, y_min); x_min = max(0, x_min)
    y_max = min(img_data.shape[0], y_max); x_max = min(img_data.shape[1], x_max)
    cropped = img_data[y_min:y_max, x_min:x_max]
    cropped_wcs = wcs_coord.slice((slice(y_min, y_max), slice(x_min, x_max)))
    return cropped, cropped_wcs


# ---------------------------
# Example usage block
# ---------------------------
if __name__ == '__main__':
    # === Update these paths to your local files before running ===
    path_radio = '/home/abdul-gani/LOFAR_plots/temp/selfcal_tec_improv_001-MFS-image-pb.fits'
    path_radio_inter = '/home/abdul-gani/LOFAR_plots/temp/Intermediate_Resol_image-MFS-image-pb.fits'
    path_residual = '/home/abdul-gani/LOFAR_plots/temp/Low_Resolution_Taper30_New-MFS-image-pb.fits'
    path_xray = '/home/abdul-gani/Xray-image.fits'

    # Load images
    radio_data, radio_wcs = loadfits(path_radio, 0)
    radio_data_inter, radio_wcs_inter = loadfits(path_radio_inter, 0)
    residual_data, residual_wcs = loadfits(path_residual, 0)
    xray_data, xray_wcs = loadfits(path_xray, 0)

    # Normalize X-ray for display
    xray_norm = simple_norm(xray_data, 'sqrt', percent=99)

    # Example: compute contour levels
    try:
        contour_levels_intermediate = define_contours_from_noise(np.max(radio_data_inter), noise=9.937e-05)
    except Exception:
        contour_levels_intermediate = define_contours(radio_data_inter)

    # Crop region of interest (example RA/Dec box - update to your science target)
    ra_min, ra_max = 209.0, 208.25
    dec_max, dec_min = 77.34, 77.19
    resi_cropped, resi_cropped_wcs = crop_image(ra_min, ra_max, dec_min, dec_max, residual_data[0][0], residual_wcs)

    # Reproject X-ray to match residual image
    xray_cropped, xray_cropped_wcs = crop_image(ra_min, ra_max, dec_min, dec_max, xray_norm, xray_wcs)
    xray_reproj, footprint = reproject_interp((xray_cropped, xray_cropped_wcs), resi_cropped_wcs, shape_out=resi_cropped.shape)

    # Plot residual image with X-ray contours and intermediate radio contours overlaid
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=resi_cropped_wcs)

    norm = colors.PowerNorm(gamma=0.5, vmin=5e-6, vmax=0.09)
    im = ax.imshow(resi_cropped, origin='lower', cmap='magma', norm=norm)

    # X-ray contours
    ax.contour(xray_reproj, levels=[0.3, 0.4, 0.5, 0.6], colors='white', linewidths=2)

    # Intermediate radio contours
    ax.contour(radio_data_inter[0][0], levels=contour_levels_intermediate, colors='cyan', linewidths=0.5, transform=ax.get_transform(radio_wcs_inter))

    # Beam (example) and scalebar
    fontprops = FontProperties(size=16)
    pixel_scale_arcsec = abs(resi_cropped_wcs.pixel_scale_matrix[1, 1]) * 3600
    kpc_per_arcsec = 5.4
    scale_bar_arcsec = 500 / kpc_per_arcsec
    scale_bar_pixels = scale_bar_arcsec / pixel_scale_arcsec

    beam = Ellipse((65, 55), width=30.3563, height=30.0803, angle=-38.6, transform=ax.transData, color='white', fill=False)
    ax.add_patch(beam)

    scalebar = AnchoredSizeBar(ax.transData, scale_bar_pixels, '500 kpc', 'lower right', pad=1.4, color='white', frameon=False, size_vertical=3, prop=fontprops)
    ax.add_artist(scalebar)

    # Colorbar on top
    fig = plt.gcf()
    cbar_ax = fig.add_axes([0.126, 0.850, 0.774, 0.03])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Flux (Jy/beam)', fontsize=14)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel('Right Ascension (J2000)', fontsize=14)
    ax.set_ylabel('Declination (J2000)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    plt.show()
