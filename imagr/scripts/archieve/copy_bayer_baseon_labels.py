import os
import glob
import re
import shutil
import multiprocessing as mp

# labels_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_office/labels/shopping_imagr_2112"
origin_bayer_dir = "/home/walter/big_daddy/nigel/modular_data_collection/ams/ams_od_0123/object_detection_ams_13_01_23/ams_collections_20230117"
target_bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/ams_office/bayer/shop_20230117"

# labels = glob.glob(f"{labels_dir}/*.txt")


def cp_mp_on_label(label):
    label_basename = os.path.basename(label)
    bayer_filename = re.sub(".bayer_8.txt", ".bayer_8", label_basename)
    bayer_original_path = os.path.join(origin_bayer_dir, bayer_filename)
    bayer_target_path = os.path.join(target_bayer_dir, bayer_filename)
    shutil.copy(bayer_original_path, bayer_target_path)


def just_cp(src, dst):
    shutil.copy(src, dst)


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

bayers = glob.glob(f"{origin_bayer_dir}/*.bayer_8")
print(len(bayers))

for bayer in bayers:
    basename = os.path.basename(bayer)
    dst = os.path.join(target_bayer_dir, basename)
    pool.apply_async(just_cp, args=(bayer, dst))

pool.close()
pool.join()
