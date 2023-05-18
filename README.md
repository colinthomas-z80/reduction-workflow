# Image Reduction Workflow
This is an attempt to put together a simple image reduction pipeline/workflow that may be ported to a workflow execution language. Most if not all 
code and information is taken from: https://github.com/astropy/ccd-reduction-and-photometry-guide

Each pair \*.py and \*\.combine.py will create a directory; \*\_reduced. 

Be warned that the LFC-DATA directory is nearly 400MB

## Dependencies
astropy ccdproc matplotlib

## Process
bias.py -> bias\_combine.py -> dark.py -> dark\_combine.py -> flats.py -> flats\_combine.py -> science.py




