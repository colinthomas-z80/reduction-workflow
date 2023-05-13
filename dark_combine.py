from pathlib import Path
import os

from astropy.stats import mad_std
import ccdproc as ccdp
import numpy as np

calibrated_path = Path("darks_reduced")
reduced_images = ccdp.ImageFileCollection(calibrated_path)

# there are multiple exposure times available. We only want to combine images with the same exposure time
darks = reduced_images.summary["imagetyp"] == "DARK"
dark_times = set(reduced_images.summary["exptime"][darks])

for exp_time in sorted(dark_times):
    calibrated_darks = reduced_images.files_filtered(imagetyp="dark", exptime=exp_time, include_path=True)

    combined_dark = ccdp.combine(calibrated_darks, method="average", 
                                    sigma_clip=True, 
                                    sigma_clip_low_thresh=5, 
                                    sigma_clip_high_thresh=5, 
                                    sigma_clip_func=np.ma.median, 
                                    sigma_clip_dev_func=mad_std, 
                                    mem_limit=350e6
                                )

    combined_dark.meta["combined"] = True

    dark_file_name = "combined_dark_{:6.3f}.fit".format(exp_time)
    combined_dark.write(calibrated_path / dark_file_name)
