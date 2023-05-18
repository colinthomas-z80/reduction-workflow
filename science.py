from pathlib import Path

import numpy as np

from astropy import units as u
import ccdproc as ccdp

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

