#!/usr/bin/env zsh

set -e

mamba env create -n mb -f environment.yaml
. `conda info --base`/etc/profile.d/conda.sh
. `conda info --base`/etc/profile.d/mamba.sh
mamba activate mb

local env_path=$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=$PWD" > $env_path
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/LuoLib/third-party/MONAI

echo >> $env_path
local nnUNet_data=$PWD/nnUNet_data
echo "export nnUNet_data=$nnUNet_data" >> $env_path
echo "export nnUNet_raw=\$nnUNet_data/raw" >> $env_path
echo "export nnUNet_preprocessed=\$nnUNet_data/preprocessed" >> $env_path
echo "export nnUNet_results=\$nnUNet_data/results" >> $env_path
