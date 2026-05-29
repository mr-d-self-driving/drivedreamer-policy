import os
import json
import pickle
import argparse

import numpy as np
import torch
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="mini")
parser.add_argument("--data_root", default="navsim_dataset")
parser.add_argument("--datalist", default=None)
parser.add_argument("--meta_dir", default=None)
parser.add_argument("--max_samples", type=int, default=10000)
args = parser.parse_args()

if args.datalist is None:
    args.datalist = f"{args.split}_meta.json"
if args.meta_dir is None:
    args.meta_dir = os.path.join(args.data_root, "meta", args.split)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
model = model.to(device=device)

with open(args.datalist, 'rb') as f:
    datas = json.load(f)

np.random.seed(2026)
np.random.shuffle(datas)

for data_n in tqdm(datas[:args.max_samples]):
    data_dir = os.path.join(args.meta_dir, data_n + '.pkl')
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    glo_images = data['glo_images']
    images = [
        glo_images['cam_f0']['image_paths'][3],
        glo_images['cam_l0']['image_paths'][3],
        glo_images['cam_r0']['image_paths'][3],
    ]
    keys = ['cam_f0', 'cam_l0', 'cam_r0']
    model.inference(
        images,
        process_res=252,
        export_dir=(images, data_dir, keys, data_n),
        export_format="depth_vis",
    )
