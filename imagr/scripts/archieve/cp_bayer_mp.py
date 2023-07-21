import os
import multiprocessing as mp
import shutil
import glob
import re


def cp_file(src, dst):
    if not os.path.exists(dst):
        shutil.copy(src, dst)


def cp_files_mp(files, dst_dir):
    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    for file in files:
        bayer_name = os.path.basename(file)
        dst = os.path.join(dst_dir, bayer_name)
        pool.apply_async(cp_file, args=(file, dst))

    # skip some frame
    # for i in range(0, len(files), 5):
    #     bayer_name = os.path.basename(files[i])
    #     dst = os.path.join(dst_dir, bayer_name)
    #     pool.apply_async(cp_file, args=(files[i], dst))

    pool.close()
    pool.join()


src_dir = "/home/walter/big_daddy/nigel/mod_od_datasets/imagr_hands_faces_170322"
dst_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/bayer/imagr_hands_faces_170322"
os.makedirs(dst_dir, exist_ok=True)

files = glob.glob(f"{src_dir}/*.bayer_8")
# files = glob.glob(f"{src_dir}/*.jpg")
print("total files number: ", len(files))
cp_files_mp(files, dst_dir)
