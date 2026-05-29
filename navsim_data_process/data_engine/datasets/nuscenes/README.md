# NuScenes Setup Guide

This module provides integration with the NuScenes dataset visualization capabilities.

This module provides guidance for setting up and visualizing results on the NuScenes dataset.

## Setup Instructions

### 1. Create and Activate Conda Environment
```bash
conda create --name nuscenes python=3.9 -y
conda activate nuscenes
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
# Basic visualization requirements
pip install imageio tqdm 'numpy<2.0'

# NuScenes toolkit and utilities
pip install nuscenes-devkit p_tqdm prettytable

# MMDetection ecosystem
pip install openmim
mim install mmcv-full
mim install 'mmdet<3.0'

# Deep learning utilities
pip install openai IPython einops psutil
pip install --use-pep517 flash-attn --no-build-isolation 
```

##  Visualization

The visualization tools include:
- `nus_visualize.py`: Main script for trajectory visualization
- Support for multiple camera views
- GIF generation capabilities
