from pathlib import Path
import os

import ccdproc as ccdp

data_path = Path("LFC-DATA")
calibrated_data = Path(".", "biases_reduced")
calibrated_data.mkdir(exist_ok=True)

files = ccdp.ImageFileCollection(data_path)

# We are going to subtract ccd overscan and trim it from each image. We will then combine all bias averages to a single file

for ccd, file_name in files.ccds(imagetyp="BIAS", ccd_kwargs={"unit":"adu"}, return_fname=True):

    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, 2055:], median=True)

    ccd = ccdp.trim_image(ccd[:, :2048])

    ccd.write(calibrated_data/file_name)

