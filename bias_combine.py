from pathlib import Path
import os

from astropy.stats import mad_std
import ccdproc as ccdp
import numpy as np

calibrated_data = Path(".", "biases_reduced")

# We are going to combine all bias averages to a single file
biases = ccdp.ImageFileCollection(calibrated_data).files_filtered(imagetyp="bias", include_path=True)

combined_bias = ccdp.combine(biases, method="average", 
                                sigma_clip=True, 
                                sigma_clip_low_thresh=5, 
                                sigma_clip_high_thresh=5, 
                                sigma_clip_func=np.ma.median, 
                                sigma_clip_dev_func=mad_std, 
                                mem_limit=350e6
                            )

combined_bias.meta["combined"] = True

combined_bias.write(calibrated_data / 'combined_bias.fit')
