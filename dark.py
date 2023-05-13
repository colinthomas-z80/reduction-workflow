from pathlib import Path
import os

import ccdproc as ccdp

data_path = Path("LFC-DATA")
darks_path = Path(data_path / "darks")

calibrated_data = Path(".", "darks_reduced")
calibrated_data.mkdir(exist_ok=True)

files = ccdp.ImageFileCollection(darks_path)

# We are going to subtract ccd overscan and trim it from each image. We will then combine all dark averages to a single file

for ccd, file_name in files.ccds(imagetyp="DARK", ccd_kwargs={"unit":"adu"}, return_fname=True):

    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[:, 2055:], median=True)

    ccd = ccdp.trim_image(ccd[:, :2048])

    ccd.write(calibrated_data / file_name)


