import multiprocessing as mp
pool = mp.Pool(processes=16)

import pickle
import os
import sys
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent.parent
sys.path.append(f"{project_root}/data_qa_generate/")
from data_engine.datasets.navsim.dataset_navsim import VLMNavsim
from data_engine.datasets.navsim.loaders.navsim.visualization.camera import _transform_pcs_to_images

import cv2
import numpy as np
import json

with open("/shared_disk/users/yang.zhou/mini_test_meta.json", "r") as f:
    meta_test_mini_list = json.load(f)


only_bev = 1
only_lidar_root = "/shared_disk/users/yang.zhou/navsim_bev_64_64"

mode = 'mini'
dataset = VLMNavsim(mode=mode)
only_lidar_dir = os.path.join(only_lidar_root, mode)
os.makedirs(only_lidar_dir, exist_ok=True)

gs_tar_h, gs_tar_w = 144, 256

def run(sid):
    container = dataset.get_container_in_only_bev(sid, 0, 1, meta_test_mini_list)

    if type(container) == tuple:
        print(f'skip exist: {container[1]}')
        # return
        with open(f"{only_lidar_dir}/{container[1]}-{11}.pkl", "rb") as f:
            bev = pickle.load(f)
        import pdb
        pdb.set_trace()
        white = np.all(bev == 255, axis=-1)
        mask = ~white
        cv2.imwrite(f'valid_mask_{sid}.jpg', (mask).astype(np.uint8)*255)

    if only_bev:
        frame_data = container["frame_data"][3]
        ann = frame_data.get("annotations", {})
        bevs = container['bev']
        if sid < 3:
            print(f"Sample {sid}")
            # print(f"  Total: {ann_data['num_total']}, Dynamic: {ann_data['num_dynamic']}")
            # print(f"  Vehicles: {ann_data['num_vehicles']}, Peds: {ann_data['num_pedestrians']}")
            # vis here
            frame_data = container["frame_data"][11]
            # mask_255 = bev_rgb_to_occ(bev)

            cv2.imwrite(f'mask_{sid}.jpg', bevs[11])
            img_f = frame_data["cameras"]["cam_f0"]["image_path"]
            img_l = frame_data["cameras"]["cam_l0"]["image_path"]
            img_r = frame_data["cameras"]["cam_r0"]["image_path"]
            os.system(f'cp {img_f} img_f_{sid}.jpg')
            os.system(f'cp {img_l} img_l_{sid}.jpg')
            os.system(f'cp {img_r} img_r_{sid}.jpg')

        for frame_idx in range(12):
            with open(f"{only_lidar_dir}/{container['token']}-{frame_idx}.pkl", "wb") as f:
                pickle.dump(bevs[frame_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
        # return container['token']

from tqdm import tqdm
import cv2

import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral')

for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):
    run(sid)    

# raw_list = [i for i in range(len(dataset))]

# pool = mp.Pool(processes=32)
# _ = list(tqdm(pool.imap(run, raw_list), total=len(raw_list), desc="Processing bev"))
# pool.close()
# pool.join()
