from pathlib import Path

import numpy as np

from astropy import visualization as aviz
from astropy.nddata.blocks import block_reduce
from astropy.nddata.utils import Cutout2D
from matplotlib import pyplot as plt


from astropy import units as u
import ccdproc as ccdp

def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10),
               cmap='viridis', log=False, clip=True,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.

    Parameters
    ----------
    image
        The image to show
    percl : number
        The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
    percu : number or None
        The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """
    if percu is None:
        percu = percl
        percl = 100 - percl

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" '
                         'if you provide one of them')
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)


    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
    fig_size_pix = fig.get_size_inches() * fig.dpi

    ratio = (image.shape // fig_size_pix).max()
    
    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio

    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in the
        # reduced image. We do *not* want to do this for images which are
        # masks, since their values should be zero or one.
         reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    norm = aviz.ImageNormalize(reduced_data,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               stretch=stretch, clip=clip)

    if is_mask:
        # The image is a mask in which pixels should be zero or one.
        # block_reduce may have changed some of the values, so reset here.
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)

    if show_colorbar:
    # I haven't a clue why the fraction and pad arguments below work to make
    # the colorbar the same height as the image, but they do....unless the image
    # is wider than it is tall. Sticking with this for now anyway...
    # Thanks: https://stackoverflow.com/a/26720422/3486425
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # In case someone in the future wants to improve this:
    # https://joseph-long.com/writing/colorbars/
    # https://stackoverflow.com/a/33505522/3486425
    # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    fig.savefig("img")


def find_nearest_dark_exposure(image, dark_exposure_times, tolerance=0.5):
    """
    Find the nearest exposure time of a dark frame to the exposure time of the image,
    raising an error if the difference in exposure time is more than tolerance.
    
    Parameters
    ----------
    
    image : astropy.nddata.CCDData
        Image for which a matching dark is needed.
    
    dark_exposure_times : list
        Exposure times for which there are darks.
    
    tolerance : float or ``None``, optional
        Maximum difference, in seconds, between the image and the closest dark. Set
        to ``None`` to skip the tolerance test.
    
    Returns
    -------
    
    float
        Closest dark exposure time to the image.
    """

    dark_exposures = np.array(list(dark_exposure_times))
    idx = np.argmin(np.abs(dark_exposures - image.header['exptime']))
    closest_dark_exposure = dark_exposures[idx]

    if (tolerance is not None and 
        np.abs(image.header['exptime'] - closest_dark_exposure) > tolerance):
        
        raise RuntimeError('Closest dark exposure time is {} for flat of exposure '
                           'time {}.'.format(closest_dark_exposure, image.header['exptime']))
        
    
    return closest_dark_exposure

darks_path = Path("darks_reduced")
flats_path = Path("flats_reduced")
reduced_path = Path("science_reduced")
reduced_path.mkdir(exist_ok=True)

science_imagetyp = "object"
flat_imagetyp = "flatfield"
exposure = "exptime"

darks_reduced = ccdp.ImageFileCollection(darks_path)
flats_reduced = ccdp.ImageFileCollection(flats_path)

ifc_reduced = ccdp.ImageFileCollection(reduced_path)
ifc_raw = ccdp.ImageFileCollection("LFC-DATA")

combined_darks = {ccd.header[exposure]: ccd for ccd in darks_reduced.ccds(imagetyp='dark', combined=True)}
combined_flats = {ccd.header['filter']: ccd for ccd in flats_reduced.ccds(imagetyp=flat_imagetyp, combined=True)}

all_reds = []
light_ccds = []

for light, file_name in ifc_raw.ccds(imagetyp=science_imagetyp, return_fname=True, ccd_kwargs=dict(unit='adu')):
    light_ccds.append(light)
        
    reduced = ccdp.subtract_overscan(light, overscan=light[:, 2055:], median=True)
                
    reduced = ccdp.trim_image(reduced[:, :2048])

    closest_dark = find_nearest_dark_exposure(reduced, combined_darks.keys())
    reduced = ccdp.subtract_dark(reduced, combined_darks[closest_dark],
            exposure_time=exposure, exposure_unit=u.second)
                        
    good_flat = combined_flats[reduced.header['filter']]
    reduced = ccdp.flat_correct(reduced, good_flat)
    all_reds.append(reduced)
    reduced.write(reduced_path / file_name)


fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)

for row, raw_science_image in enumerate(light_ccds):
    filt = raw_science_image.header['filter']
    axes[row, 0].set_title('Uncalibrated image {}'.format(filt))
    show_image(raw_science_image, cmap='gray', ax=axes[row, 0], fig=fig, percl=90)

    axes[row, 1].set_title('Calibrated image {}'.format(filt))
    show_image(all_reds[row].data, cmap='gray', ax=axes[row, 1], fig=fig, percl=90)
