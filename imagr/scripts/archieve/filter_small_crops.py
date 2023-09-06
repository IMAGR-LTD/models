import os 
import glob 
import shutil 
from PIL import Image 
import numpy as np
import multiprocessing as mp



def per_barcode(barcode):
    bar_dir = os.path.join(crop_dir, barcode)
    cams = os.listdir(bar_dir)
    for cam in cams:
        cam_dir = os.path.join(bar_dir, cam)
        files = glob.glob(f"{cam_dir}/*.jpg")
        for file in files:
            basename = os.path.basename(file)
            dst_big_crop = os.path.join(save_dir, barcode, cam)
            os.makedirs(dst_big_crop, exist_ok=True)
            dst_big_crop_mean = os.path.join(save_mean_dir, barcode, cam)
            os.makedirs(dst_big_crop_mean, exist_ok=True)


            img = Image.open(file)
            w,h = img.size

            if w * h > 55 * 75:
                shutil.copy(file, os.path.join(dst_big_crop, basename))
            
            img_np = np.array(img)
            if w * h > 55 * 75 and img_np.mean() > 120:
                shutil.copy(file, os.path.join(dst_big_crop_mean, basename))



crop_dir = "/home/walter/big_daddy/onboard_crops"
save_dir = "/home/walter/big_daddy/onboard_big_crops"
save_mean_dir = "/home/walter/big_daddy/onboard_big_crops_means"
os.makedirs(save_dir, exist_ok=True)

barcodes = os.listdir(crop_dir)

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

for barcode in barcodes:
    print(barcode)
    pool.apply_async(per_barcode, args=(barcode,))

pool.close()
pool.join()