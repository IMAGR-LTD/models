from email.errors import HeaderParseError
import numpy as np
import cv2
import os
import time
import multiprocessing as mp
import argparse
from pathlib import Path
import glob 
import re

def _isp_pipe(img, black_level=3074.0 / 65535.0):
    """
    ISP pipeline including black level correction, lense shading correction, EQ
    img: single channel bayer image in range 0-255
    bbox: bounding box to get the ROI from LSC_LUT, tuple of (top, bottom, left, right)
    """
    white_clip_amax = np.amax(img)
    img = np.array(img, dtype=np.float32) / 255.0
    # black level correction
    white_level = white_clip_amax / 255.0
    img = (img - black_level) / (white_level - black_level)
    img = img * white_level
    img = np.clip(img, 0.0, 1.0)
    # channel EQ @ 5000K
    img[::2, ::2] = img[::2, ::2] * 1.5
    img[1::2, 1::2] = img[1::2, 1::2] * 1.85
    img = img * 1.9
    img = np.clip(img, 0.0, 1.0)

    # cast back to 255 uint8 image
    img = np.array(img * 255, dtype=np.uint8)
    # image is actually BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    rgb_max = np.amax(img_rgb)
    clip_max = int(rgb_max * 0.92)
    clip_mask = np.sum(img_rgb > clip_max, axis=2) > 1
    clip_mask = np.stack((clip_mask,) * 3, axis=2)
    img_rgb[clip_mask] = 255
    return img_rgb


def _isp_imx462(img):
    """
    ISP in this case only consist of bayer multipliers and conversion from bayer to rgb
    img represented by float value 0-1
    """
    img = np.array(img, dtype=np.float32) / 255.0
    # red bayer channel multiplicand
    img[::2, ::2] = img[::2, ::2] * 1.5
    # blue bayer channel multiplicand
    img[1::2, 1::2] = img[1::2, 1::2] * 2.25  # was 1.85
    # global bayer multiplicand
    # img= img * 1.9 #was 1.9
    img = np.clip(img, 0.0, 1.0)
    img = np.array(img * 255, dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    # returns an image in a linear colorspace
    return img_rgb


def _apply_gamma(img, gamma=0.55):
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.power(img, gamma)
    img = img * 255.0
    return np.array(img, dtype=np.uint8)


def _process_bayer(bar, in_dir, out_dir):
    os.makedirs(os.path.join(out_dir, bar), exist_ok=True)
    for img_path in os.listdir(os.path.join(in_dir, bar)):
        
        if img_path.endswith("bayer_8"):
            img = np.fromfile(open(os.path.join(
                in_dir, bar, img_path), 'rb'), dtype=np.uint8).reshape(1080, 1920)
            img = _isp_imx462(img)
            img = _apply_gamma(img)
            _ = cv2.imwrite(os.path.join(out_dir, bar, img_path +
                            ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            

def bayer_to_jpg_mp(bayer, save_path):
    img = np.fromfile(open(bayer, 'rb'), dtype=np.uint8).reshape(1080, 1920)
    img = _isp_imx462(img)
    img = _apply_gamma(img)
    _ = cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir',action='store',help='input path to the raw bayer dir, should contain subdir for different barcodes', required=True)
    parser.add_argument('-o', '--out_dir',action='store',help='output path for the processed jpeg', required=True)
    parser.add_argument('-c', '--contain_barcodes', action='store_true', help='is the input dir contains multiple barcodes'
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)
    print(pool)

    files = glob.glob(f"{in_dir}/*.bayer_8")
    print(len(files))

    for file in files:
        basename = os.path.basename(file)
        savename = re.sub(".bayer_8", ".bayer_8.jpg", basename)
        save_path = os.path.join(out_dir, savename)
        if not os.path.exists(save_path):
            pool.apply_async(bayer_to_jpg_mp, args=(file, save_path))
    
    pool.close()
    pool.join()