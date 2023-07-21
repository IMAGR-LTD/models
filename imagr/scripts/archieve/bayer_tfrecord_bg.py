import numpy as np
import os
import multiprocessing as mp
import tensorflow as tf
import glob
import re
import argparse
from PIL import Image 
import hashlib


def bayer_to_byte(src, width=1920, height=1080):
    bayer = np.fromfile(src, np.uint8).reshape((1, height, width, 1))
    bayer = tf.nn.space_to_depth(bayer, block_size=2)
    debayered = tf.stack([
        bayer[:, :, :, 0] , #R
        bayer[:, :, :, 1] , #G
        bayer[:, :, :, 3]   #B
    ], axis=3).numpy()

    tmp = "/tmp/images"
    os.makedirs(tmp, exist_ok=True)
    basename = os.path.basename(src)
    basename = re.sub(".bayer_8", ".jpg", basename)
    dst = os.path.join(tmp, basename)
    img = tf.keras.utils.array_to_img(debayered.squeeze())
    img.save(dst)
    bytes = open(dst, 'rb').read()

    return bytes


def generate_example(bayer_path):
    """label is 1 for SSD Mobilenetv2 for item, 0 for background

    Returns:
        tf.train.Example: tfrecord example contains all the info 
        needed for training 
    """
    encoded_jpg = bayer_to_byte(bayer_path)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classes_text = []
    label = []

    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[540])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[960])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(files, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file in files:
            example = generate_example(file)
            writer.write(example.SerializeToString())


def per_folder(folder_path, tfrecord_path):
    bayer_files = glob.glob(f"{folder_path}/*.bayer_8")
    print(f"total bayer files: {len(bayer_files)}")
    splits = int(len(bayer_files) / 1000) + 1
    
    for i in range(splits):
        tfrecord_name = f"{tfrecord_path}/tfrecord_{i:04d}.tfrecord"
        start = i * 1000
        end = min(len(bayer_files), start + 1000)
        print(start, end)
        if start == end:
            continue
        write_tfrecord(bayer_files[start:end], tfrecord_name)


bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/sonae/bayer"
tfrecord_dir = re.sub("bayer", "tfrecord", bayer_dir)


folders = ["MC_shopping_trolley_OD_bgnd_210223"]

# for folder in folders:
#     folder_path = os.path.join(bayer_dir, folder)
#     tfrecord_path = os.path.join(tfrecord_dir, folder)
#     os.makedirs(tfrecord_path, exist_ok=True)
#     per_folder(folder_path, tfrecord_path)


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)

for folder in folders:
    folder_path = os.path.join(bayer_dir, folder)
    print(folder_path)
    tfrecord_path = os.path.join(tfrecord_dir, folder)
    os.makedirs(tfrecord_path, exist_ok=True)
    pool.apply_async(per_folder, args=(folder_path, tfrecord_path,))

pool.close()
pool.join()