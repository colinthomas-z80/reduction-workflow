from pathlib import Path

from astropy import units as u
from astropy.nddata import CCDData

import ccdproc as ccdp
import numpy as np

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


bias_path = Path("biases_reduced")
dark_path = Path("darks_reduced")
flat_path = Path("flats_reduced")
flat_path.mkdir(exist_ok=True)

ifc_reduced = ccdp.ImageFileCollection(dark_path)
combined_dark_files = {ccd.header["exptime"]: ccd for ccd in ifc_reduced.ccds(imagetyp="dark", combined=True)}

actual_exposure_times = set(h['exptime'] for h in ifc_reduced.headers(imagetyp='dark', combined=True))

flat_image_type = "FLATFIELD"

raw_data = Path("LFC-DATA")
ifc_raw = ccdp.ImageFileCollection(raw_data)

# Our darks and flats have nearly the same exposure times, so we do not need to scale, nor subtract bias

for ccd, file_name in ifc_raw.ccds(imagetyp="FLATFIELD", ccd_kwargs={"unit":"adu"}, return_fname=True):

    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, :2055], median=True)

    ccd = ccdp.trim_image(ccd[:, :2048])

    closest_dark = find_nearest_dark_exposure(ccd, actual_exposure_times)
    
    ccd = ccdp.subtract_dark(ccd, combined_dark_files[closest_dark], exposure_time="exptime", exposure_unit=u.second)

    ccd.write(flat_path / ("flat-" + file_name))

