import pickle
import json
import os
import argparse
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2

from typing import List

import numpy as np
import numpy.typing as npt

# we add an idx to the original func.
# from nuplan.common.geometry.convert import absolute_to_relative_poses

from nuplan.common.geometry.convert import pose_from_matrix, matrix_from_pose


def absolute_to_relative_poses(absolute_poses: List[StateSE2], idx=0) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from absolute to relative coordinates with the first pose being the origin
    :param absolute_poses: list of absolute poses to convert
    :return: list of converted relative poses
    """
    absolute_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in absolute_poses])
    origin_transform = np.linalg.inv(absolute_transforms[idx])
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = [pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms]

    return relative_poses


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="mini")
parser.add_argument("--data_root", default="navsim_dataset")
args = parser.parse_args()

split = args.split
data_root = args.data_root
datalist_path = f'{split}_meta.json'
with open(datalist_path, "rb") as f:
    raw_list = json.load(f)

base_dir = os.path.join(data_root, 'meta', split)
video_dir = os.path.join(data_root, 'navsim_video', split)

from PIL import Image
from moviepy.editor import ImageSequenceClip
def images_to_video(image_folder, output_video, fps=20):
    # images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
    # images.sort()  # Ensure the images are in the correct order
    images = image_folder
    # images.sort(key=lambda p: int(p.split('/')[-1].split('.')[0].split('_')[-1]))

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_video, codec="libx264",
        verbose=False,   # 关闭 MoviePy 自己的进度条
        logger=None      # 禁止写自定义 logger（否则仍会输出）
    )


import numpy as np

# single
def run(raw):
    data_path = os.path.join(base_dir, f'{raw}.pkl')
    try:
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
    except:
        print(f'[{raw}] loading pkl fail')
        return (0, raw)

    glo_pose = raw_data['glo_status']['global_poses']
    ego_pose_se2 = [StateSE2(float(x), float(y), float(yaw)) for x, y, yaw in glo_pose[:12]]
    rel_ego_pose_se2 = absolute_to_relative_poses(ego_pose_se2, 3)
    rel_ego_pose_se2 = [[pose.x, pose.y, pose.heading] for pose in rel_ego_pose_se2]
    ego_pose = np.array(rel_ego_pose_se2)
    return (ego_pose[4:, :2] - ego_pose[3:4, :2])


# for raw in tqdm(raw_list):
#     traj = run(raw)
#     import pdb
#     pdb.set_trace()

import multiprocessing as mp
pool = mp.Pool(processes=16)
all_clips = list(tqdm(pool.imap(run, raw_list), total=len(raw_list), desc="Processing Videos"))
pool.close()
pool.join()

dataset_traj = np.array(all_clips)

all_x = dataset_traj[..., 0]
all_y = dataset_traj[..., 1]

x_mean = all_x.mean()
x_std = all_x.std() + 1e-6 # 加一点epsilon防除零

y_mean = all_y.mean()
y_std = all_y.std() + 1e-6

print(x_mean, x_std)
print(y_mean, y_std)

# 10.172484, 8.805105
# 0.360722, 2.277741