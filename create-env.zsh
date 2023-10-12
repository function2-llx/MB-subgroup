#!/usr/bin/env zsh

set -e

mamba env create -n mb -f environment.yaml
. `conda info --base`/etc/profile.d/conda.sh
. `conda info --base`/etc/profile.d/mamba.sh
mamba activate mb
echo "export PYTHONPATH=$PWD:\$PYTHONPATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/LuoLib/third-party/MONAI
