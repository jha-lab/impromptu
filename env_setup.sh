#!/bin/sh

# module load anaconda3/2020.11
conda create --name impromptu pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia

conda activate impromptu

# Add other packages and enabling extentions
conda install -c conda-forge tqdm ipywidgets matplotlib scikit-optimize
jupyter nbextension enable --py widgetsnbextension

#Install tensorboard
conda install -c conda-forge tensorboard
