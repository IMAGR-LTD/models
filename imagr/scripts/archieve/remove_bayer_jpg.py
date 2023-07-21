import os
import glob
import re
import multiprocessing as mp
import shutil


def remove_no_label_bayers(labels, bayers):
    for bayer in bayers:
        label_file = re.sub(".bayer_8", ".bayer_8.txt", bayer)
        label_file = re.sub("/bayer", "/labels", label_file)
        if not label_file in labels:
            os.remove(bayer)


def remove_no_label_jpgs(labels, jpgs):
    for jpg in jpgs:
        label_file = re.sub(".jpeg", ".txt", jpg)
        label_file = re.sub("/images", "/labels", label_file)
        if not label_file in labels:
            os.remove(jpg)


def mv_file(src, dst):
    if not os.path.exists(dst):
        shutil.move(src, dst)


def mv_files_mp(files, dst_dir):
    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    for file in files:
        bayer_name = os.path.basename(file)
        dst = os.path.join(dst_dir, bayer_name)
        pool.apply_async(mv_file, args=(file, dst))

    pool.close()
    pool.join()


def count_no_label_bayer(labels, bayers):
    count = 0
    for bayer in bayers:
        label_file = re.sub(".bayer_8", ".bayer_8.txt", bayer)
        label_file = re.sub("/bayer", "/labels", label_file)
        if not label_file in labels:
            count += 1
    print("no label bayers count: ", count)


def count_no_label_jpg(labels, jpgs):
    count = 0
    for jpg in jpgs:
        if jpg.split(".")[-1] == "jpg":
            label_file = re.sub(".jpg", ".txt", jpg)
        elif jpg.split(".")[-1] == "jpeg":
            label_file = re.sub(".jpeg", ".txt", jpg)
        else:
            print("illeage filename")
            continue
        
        label_file = re.sub("/images", "/labels", label_file)
        if not label_file in labels:
            count += 1
    print("no label jpgs count: ", count)


def move_no_label_bayers(labels, bayers, dst):
    to_move_list = []
    for bayer in bayers:
        label_file = re.sub(".bayer_8", ".bayer_8.txt", bayer)
        label_file = re.sub("/bayer", "/labels", label_file)
        if not label_file in labels:
            to_move_list.append(bayer)

    mv_files_mp(to_move_list, dst)


def move_no_label_jpg(labels, jpgs, dst):
    to_move_list = []
    for jpg in jpgs:
        if jpg.split(".")[-1] == "jpg":
            label_file = re.sub(".jpg", ".txt", jpg)
        elif jpg.split(".")[-1] == "jpeg":
            label_file = re.sub(".jpeg", ".txt", jpg)
        else:
            print("illeage filename")
            continue

        
        label_file = re.sub("/images", "/labels", label_file)
        if not label_file in labels:
            to_move_list.append(jpg)

    mv_files_mp(to_move_list, dst)


label_path = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae/labels/onboard_finetune_v3"
bayer_path = re.sub("/labels", "/bayer", label_path)
jpgs_path = re.sub("/labels", "/images", label_path)

labels = glob.glob(f"{label_path}/*.txt")
bayers = glob.glob(f"{bayer_path}/*.bayer_8")
jpgs = glob.glob(f"{jpgs_path}/*")

print("num of label files: ", len(labels))
print("num of bayer files: ", len(bayers))
print("num of jpg files: ", len(jpgs))

# count_no_label_bayer(labels, bayers)
# count_no_label_jpg(labels, jpgs)

dst = re.sub("/labels", "/del_images", label_path)
os.makedirs(dst, exist_ok=True)
# move_no_label_bayers(labels, bayers, dst)
move_no_label_jpg(labels, jpgs, dst)


