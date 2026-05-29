# import pickle
# import json
# import os
# from tqdm import tqdm

# from nuplan.common.actor_state.state_representation import StateSE2
# from nuplan.common.geometry.convert import absolute_to_relative_poses

# datalist_path = '/shared_disk/users/yang.zhou/train_meta.json'
# with open(datalist_path, "rb") as f:
#     raw_list = json.load(f)

# base_dir = '/shared_disk/users/yang.zhou/navsim_dataset/meta/train'
# video_dir = "/shared_disk/users/yang.zhou/navsim_video/train"

# from PIL import Image
# from moviepy.editor import ImageSequenceClip
# def images_to_video(image_folder, output_video, fps=20):
#     # images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
#     # images.sort()  # Ensure the images are in the correct order
#     images = image_folder
#     # images.sort(key=lambda p: int(p.split('/')[-1].split('.')[0].split('_')[-1]))

#     clip = ImageSequenceClip(images, fps=fps)
#     clip.write_videofile(output_video, codec="libx264",
#         verbose=False,   # 关闭 MoviePy 自己的进度条
#         logger=None      # 禁止写自定义 logger（否则仍会输出）
#     )


# import numpy as np

# # single
# def run(raw):
#     data_path = os.path.join(base_dir, f'{raw}.pkl')
#     try:
#         with open(data_path, "rb") as f:
#             raw_data = pickle.load(f)
#     except:
#         print(f'[{raw}] loading pkl fail')
#         return (0, raw)
#     # glo_images = raw_data['glo_images']
#     # all_good = 0
#     # for view in ['cam_f0', 'cam_l0', 'cam_r0']:
#     #     try:
#     #         img_list = glo_images[view]['image_paths'][3:12]
#     #         video_view_dir = os.path.join(video_dir, view)
#     #         if os.path.exists(os.path.join(video_view_dir, raw+'.mp4')):
#     #             all_good += 1
#     #             continue
#     #         os.makedirs(video_view_dir, exist_ok=True)
#     #         images_to_video(img_list, os.path.join(video_view_dir, raw+'.mp4'), fps=2)
#     #         ext = img_list[0][-4:]
#     #         # os.system(f'cp {img_list[0]} {os.path.join(video_view_dir, raw+ext)}')
#     #         all_good += 1
#     #     except Exception as e:
#     #         # print(glo_images[view]['image_paths'][3:12])
#     #         print(f'[{raw}_{view}] data missing: {e}')
#     # return (all_good, raw)

#     glo_pose = raw_data['glo_status']['global_poses']
#     ego_pose_se2 = [StateSE2(float(x), float(y), float(yaw)) for x, y, yaw in glo_pose[:12]]
#     rel_ego_pose_se2 = absolute_to_relative_poses(ego_pose_se2, 3)
#     rel_ego_pose_se2 = [[pose.x, pose.y, pose.heading] for pose in rel_ego_pose_se2]
#     ego_pose = np.array(rel_ego_pose_se2)
#     return (ego_pose[4:, :2] - ego_pose[3:4, :2])


# # for raw in tqdm(raw_list):
# #     traj = run(raw)
# #     import pdb
# #     pdb.set_trace()

# import multiprocessing as mp
# pool = mp.Pool(processes=16)
# all_clips = list(tqdm(pool.imap(run, raw_list), total=len(raw_list), desc="Processing Videos"))
# pool.close()
# pool.join()

# dataset_traj = np.array(all_clips)

# all_x = dataset_traj[..., 0]
# all_y = dataset_traj[..., 1]

# x_mean = all_x.mean()
# x_std = all_x.std() + 1e-6 # 加一点epsilon防除零

# y_mean = all_y.mean()
# y_std = all_y.std() + 1e-6

# print(x_mean, x_std)
# print(y_mean, y_std)
# import pdb
# pdb.set_trace()

# # 10.172484, 8.805105
# # 0.360722, 2.277741



import pickle
import json
import os
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses

datalist_path = '/shared_disk/users/yang.zhou/waymo-e2e-original_train_meta.json'
with open(datalist_path, "rb") as f:
    raw_list = json.load(f)

base_dir = '/shared_disk/users/yang.zhou/waymo_e2e/processed/waymoe2e_train_meta'
# video_dir = "/shared_disk/users/yang.zhou/navsim_video/train"

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
    # glo_images = raw_data['glo_images']
    # all_good = 0
    # for view in ['cam_f0', 'cam_l0', 'cam_r0']:
    #     try:
    #         img_list = glo_images[view]['image_paths'][3:12]
    #         video_view_dir = os.path.join(video_dir, view)
    #         if os.path.exists(os.path.join(video_view_dir, raw+'.mp4')):
    #             all_good += 1
    #             continue
    #         os.makedirs(video_view_dir, exist_ok=True)
    #         images_to_video(img_list, os.path.join(video_view_dir, raw+'.mp4'), fps=2)
    #         ext = img_list[0][-4:]
    #         # os.system(f'cp {img_list[0]} {os.path.join(video_view_dir, raw+ext)}')
    #         all_good += 1
    #     except Exception as e:
    #         # print(glo_images[view]['image_paths'][3:12])
    #         print(f'[{raw}_{view}] data missing: {e}')
    # return (all_good, raw)

    glo_pose = raw_data['glo_status']['future_poses']
    # ego_pose_se2 = [StateSE2(float(x), float(y), float(yaw)) for x, y, yaw in glo_pose[:12]]
    # rel_ego_pose_se2 = absolute_to_relative_poses(ego_pose_se2, 3)
    # rel_ego_pose_se2 = [[pose.x, pose.y, pose.heading] for pose in rel_ego_pose_se2]
    return glo_pose


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
import pdb
pdb.set_trace()

# 10.172484, 8.805105
# 0.360722, 2.277741

# 14.89949 17.638371513916017
# -0.038201112 2.660390184951782
