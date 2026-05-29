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


only_lidar = 1
only_lidar_root = "/shared_disk/users/yang.zhou/navsim_dataset/meta"

mode = 'test'
dataset = VLMNavsim(mode=mode)
only_lidar_dir = os.path.join(only_lidar_root, mode)
os.makedirs(only_lidar_dir, exist_ok=True)

gs_tar_h, gs_tar_w = 144, 256

def run(sid):
    container = dataset.get_container_in_only_lidar(sid, only_lidar, 0, meta_test_mini_list)

    if only_lidar:
        depth = {}
        # only consider 3 so 0 is idx
        lidar_idx = 3
        lidar = container['frame_data'][lidar_idx]['lidar']
        for view in ['cam_f0', 'cam_l0', 'cam_r0']:
                r = container["images"][lidar_idx][view]["sensor2lidar_rotation"]
                t = container["images"][lidar_idx][view]["sensor2lidar_translation"]
                k = container["images"][lidar_idx][view]["intrinsics"]
                # img_h, img_w = 1080, 1920

                img_path = container["images"][lidar_idx][view]["image_path"]  # 这个key名你得用下面print确认
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]

                pixel_coords, valid_mask, depth_values = _transform_pcs_to_images(lidar, r, t, k, (img_h, img_w), (gs_tar_h, gs_tar_w))

                # Get valid depth points and corresponding coordinates
                valid_depth_points = depth_values[valid_mask]
                valid_cam_coords = pixel_coords[valid_mask]
                # Convert coordinates to integer indices
                x_indices = valid_cam_coords[:, 0].astype(np.int32)
                y_indices = valid_cam_coords[:, 1].astype(np.int32)
                # Initialize arrays to accumulate depth sums and counts
                depth_sums = np.zeros((gs_tar_h, gs_tar_w))
                depth_counts = np.zeros((gs_tar_h, gs_tar_w))

                np.add.at(depth_sums, (y_indices, x_indices), valid_depth_points)
                np.add.at(depth_counts, (y_indices, x_indices), 1)

                depth_map = np.divide(depth_sums, depth_counts, where=depth_counts > 0)

                depth_map[depth_counts == 0] = 0

                depth[view] = depth_map

        with open(f"{only_lidar_dir}/{container['token']}.pkl-depth_gt.pkl", "wb") as f:
            pickle.dump(depth, f, protocol=pickle.HIGHEST_PROTOCOL)
        return depth, container['token']

from tqdm import tqdm
import cv2

import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral')

# for sid in tqdm(range(len(dataset)), desc=f"Processing {mode} samples"):
#     depths, tok = run(sid)
#     ## da3
#     # da3_path = os.path.join('/shared_disk/users/yang.zhou/navsim_dataset/meta', mode, tok+'.pkl-depth.pkl')
#     # with open(da3_path, "rb") as f:
#     #     da3_depth = pickle.load(f)
    
#     pkl_path = os.path.join('/shared_disk/users/yang.zhou/navsim_dataset/meta', mode, tok+'.pkl')

#     with open(pkl_path, "rb") as f:
#         meta_data = pickle.load(f)
#     imgs = meta_data['glo_images']

#     for view in ['cam_l0', 'cam_f0', 'cam_r0']:
#         depth = depths[view]
#         import pdb
#         pdb.set_trace()

#         filename = imgs[view]['image_paths'][3]

#         # da3_d = da3_depth[view]

#         # da3_d = cv2.resize(da3_d, (256, 144), interpolation=cv2.INTER_NEAREST)

#         image = cv2.imread(filename)
#         H, W = image.shape[:2]
#         image = cv2.resize(image, (256, 144))  # (width, height)


#         vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#         vis_depth = vis_depth.astype(np.uint8)
#         vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

#         # vis_depth_ = (da3_d - da3_d.min()) / (da3_d.max() - da3_d.min()) * 255.0
#         # vis_depth_ = vis_depth_.astype(np.uint8)
#         # vis_depth_ = (cmap(vis_depth_)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

#         split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
#         # combined_result = cv2.hconcat([image, split_region, vis_depth, split_region, vis_depth_])
#         combined_result = cv2.hconcat([image, split_region, vis_depth])
#         cv2.imwrite(f'./{tok}_{view}'+ '.png', combined_result)
#         import pdb
#         pdb.set_trace()

        

raw_list = [i for i in range(len(dataset))]

pool = mp.Pool(processes=16)
_ = list(tqdm(pool.imap(run, raw_list), total=len(raw_list), desc="Processing lidar"))
pool.close()
pool.join()
