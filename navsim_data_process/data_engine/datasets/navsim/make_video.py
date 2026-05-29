import pickle
import json
import os
from tqdm import tqdm

datalist_path = '/shared_disk/users/yang.zhou/train_meta.json'
with open(datalist_path, "rb") as f:
    raw_list = json.load(f)

base_dir = '/shared_disk/users/yang.zhou/navsim_dataset/meta/train'
video_dir = "/shared_disk/users/yang.zhou/navsim_video/train"

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

# single
def run(raw):
    data_path = os.path.join(base_dir, f'{raw}.pkl')
    try:
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
    except:
        print(f'[{raw}] loading pkl fail')
        return (0, raw)
    glo_images = raw_data['glo_images']
    all_good = 0
    for view in ['cam_f0', 'cam_l0', 'cam_r0']:
        try:
            img_list = glo_images[view]['image_paths'][3:12]
            video_view_dir = os.path.join(video_dir, view)
            if os.path.exists(os.path.join(video_view_dir, raw+'.mp4')):
                all_good += 1
                continue
            os.makedirs(video_view_dir, exist_ok=True)
            images_to_video(img_list, os.path.join(video_view_dir, raw+'.mp4'), fps=2)
            ext = img_list[0][-4:]
            # os.system(f'cp {img_list[0]} {os.path.join(video_view_dir, raw+ext)}')
            all_good += 1
        except Exception as e:
            # print(glo_images[view]['image_paths'][3:12])
            print(f'[{raw}_{view}] data missing: {e}')
    return (all_good, raw)

# for raw in tqdm(raw_list):
#     run(raw)

import multiprocessing as mp
pool = mp.Pool(processes=16)
all_clips = list(tqdm(pool.imap(run, raw_list), total=len(raw_list), desc="Processing Videos"))
pool.close()
pool.join()

count = 0
for clip in all_clips:
    if clip[0] != 3:
        print(f'{clip[1]} fail in video data')
        count += 1

print(count)

