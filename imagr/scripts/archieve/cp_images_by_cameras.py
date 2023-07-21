import os 
import multiprocessing as mp
import shutil
import glob
import re
from pathlib import Path


def cp_image(src, dst):
    shutil.copy(src, dst)


images_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/cvat_data/sonae_shop/images/sonae_shop_3"
# cameras = ["0104", "0105", "0106", "0156", "0114", "0111", "0113", "0197", "0196"]
cameras = ["0101", "0102", "0103"]





dataset = os.path.basename(images_dir)
for camera in cameras:
    parent_path = Path(images_dir).parent.absolute()
    out_dir = os.path.join(parent_path, f"{dataset}_{camera}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir for camera {camera} is ", out_dir)


images = glob.glob(f"{images_dir}/*.jpg")
print("total images in the current dir is ", len(images))


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)


for image in images:
    basename = os.path.basename(image)
    for camera in cameras:
        if f"akl-{camera}" in image:
            out_dir = os.path.join(parent_path, f"{dataset}_{camera}")
            dst = os.path.join(out_dir, basename)
            pool.apply_async(cp_image, args=(image, dst))
    

pool.close()
pool.join()