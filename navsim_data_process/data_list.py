import os
import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="mini")
parser.add_argument("--data_root", default="navsim_dataset")
args = parser.parse_args()

root = os.path.join(args.data_root, "meta", args.split)

res = []
for f in os.listdir(root):
    if f.endswith('.pkl') and not f.endswith('-depth.pkl'):
        res.append(f[:-4])

random.seed(42)
random.shuffle(res)

out_path = f'{args.split}_meta.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4)
print(f"Saved {len(res)} samples to {out_path}")
