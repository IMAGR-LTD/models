import os
import multiprocessing as mp
import glob
import re
import shutil

def per_barcode(barcode):
    camera_ids = set()
    barcode_dir = os.path.join(src_dir, barcode)
    files = glob.glob(f"{barcode_dir}/*.jpeg")
    for file in files:
        file_basename = os.path.basename(file)
        camera_id = file_basename.split("-")[4].split(":")[0]
        camera_ids.add(camera_id)

    for camera_id in camera_ids:
        files = glob.glob(f"{barcode_dir}/*{camera_id}*")
        num_files = len(files)
        if num_files < 20:
            for file in files:
                new_filename = os.path.join(dst, os.path.basename(file))
                shutil.copy(file, new_filename)
        elif num_files < 30:
            for i in range(0, num_files, 2):
                new_filename = os.path.join(dst, os.path.basename(files[i]))
                shutil.copy(files[i], new_filename)
        else:
            for i in range(0, num_files, 3):
                new_filename = os.path.join(dst, os.path.basename(files[i]))
                shutil.copy(files[i], new_filename)

    print("finish cp for barcode ", barcode)

src_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/cvat_data/ams_onboard_finetune/images"
dst = "/home/walter/nas_cv/walter_stuff/modular_dataset/cvat_data/ams_onboard_finetune/relabel"
os.makedirs(dst, exist_ok=True)

barcodes = os.listdir(src_dir)
print(len(barcodes))


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)

for barcode in barcodes:
    pool.apply_async(per_barcode, args=(barcode,))

pool.close()
pool.join()

