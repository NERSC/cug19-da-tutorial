#!/bin/bash

# This is just an example script of how to install the kernel for jupyter.
# Attendees are not expected to run this script.

# Setup the python installation
. /usr/common/software/python/3.6-anaconda-5.2/etc/profile.d/conda.sh
conda activate /global/cscratch1/sd/sfarrell/conda/cug19

# To install the kernel to your user directory:
python -m ipykernel install --user --name cug19-tutorial

# To install centrally on Cori JupyterHub (run as swowner, not for users):
#umask 002
#python -m ipykernel install --prefix /global/common/cori/software/python/3.6-anaconda-5.2 --name cug19-tutorial
