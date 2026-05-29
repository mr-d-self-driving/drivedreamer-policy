
# NAVSIM Setup Guide

This module provides guidance for setting up on the NAVSIM dataset.


## Setup Instructions

### 1. Create and Activate Conda Environment
```bash
conda create --name driveemma_navsim python=3.9 -y
conda activate driveemma_nuscenes
```

### 2. Install PyTorch
Install the latest compatible version of PyTorch based on your system and CUDA version:

Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select your environment configuration.

Example installation:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install Additional Dependencies
```bash
# Core ML dependencies
pip install tqdm hydra-core lightning
pip install pyquaternion geopandas
pip install nuscenes-devkit

# Visualization tools
pip install imageio opencv-python matplotlib

# Additional utilities
pip install aioboto3 pytest rasterio
pip install prettytable retry psutil
pip install shapely ray IPython openai
pip install einops
pip install --use-pep517 flash-attn --no-build-isolation
pip install 'numpy<2.0'

# Install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit
cd nuplan-devkit
pip install .
cd ..
```

### 4. Install Navsim Datasets
NavSim:https://github.com/autonomousvision/navsim
The NavSim dataset is located under the path `data_qa_generate/data_engine/data_storage/external_datasets/navsim`.

### 5. Generate Navsim Data_QA
python data_qa_generate/data_engine/datasets/navsim/q7_navsim.py
python data_qa_generate/data_engine/datasets/navsim/navsim_qa.py
