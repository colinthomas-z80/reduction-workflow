
from pathlib import Path
import os

from astropy.stats import mad_std
import ccdproc as ccdp
import numpy as np

calibrated_path = Path("flats_reduced")
reduced_images = ccdp.ImageFileCollection(calibrated_path)

flat_filters = set(h["filter"] for h in reduced_images.headers(imagetyp="flatfield"))

def inv_median(a):
    return 1 / np.median(a)

# there are multiple exposure times available. We only want to combine images with the same exposure time
darks = reduced_images.summary["imagetyp"] == "DARK"
dark_times = set(reduced_images.summary["exptime"][darks])

for filt in flat_filters:
    to_combine = reduced_images.files_filtered(imagetyp="flatfield", filter=filt, include_path=True)
            
    combined_flat = ccdp.combine(to_combine,
        method='average', scale=inv_median,
        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
        mem_limit=350e6
        )

    combined_flat.meta['combined'] = True
    dark_file_name = 'combined_flat_filter_{}.fit'.format(filt.replace("''", "p"))
    combined_flat.write(calibrated_path / dark_file_name)
