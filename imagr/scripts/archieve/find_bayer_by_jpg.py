import os 
import shutil 
import glob
import re

img_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/images/shop_multi_classes"
bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_office/bayer/shopping_imagr_2112"
dst_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/bayer/shop_multi_classes"
os.makedirs(dst_dir, exist_ok=True)

imgs = glob.glob(f"{img_dir}/*.jpg")
for img in imgs:
    img_basename = os.path.basename(img)
    bayer_basename = re.sub(".bayer_8.jpg", ".bayer_8", img_basename)
    bayer_src = os.path.join(bayer_dir, bayer_basename)
    bayer_dst = os.path.join(dst_dir, bayer_basename)
    shutil.copy(bayer_src, bayer_dst)