import os
import io
import tensorflow as tf
import hashlib
import re
import glob
import numpy as np
import multiprocessing as mp


def bayer_to_byte(bayer_path, image_savedir, width=1920, height=1080):
    bayer = np.fromfile(bayer_path, np.uint8).reshape((1, height, width, 1))
    bayer = tf.nn.space_to_depth(bayer, block_size=2)
    debayered = tf.stack([
        bayer[:, :, :, 0] , #R
        bayer[:, :, :, 1] , #G
        bayer[:, :, :, 3]   #B
    ], axis=3).numpy()

    basename = os.path.basename(bayer_path)
    basename = re.sub(".bayer_8", ".bayer_8.jpg", basename)
    dst = os.path.join(image_savedir, basename)
    img = tf.keras.utils.array_to_img(debayered.squeeze())
    img.save(dst)
    bytes = open(dst, 'rb').read()

    return bytes

example_count = 0
def create_single_example(bayer_path, annos, image_dir):
    """label is 1 for SSD Mobilenetv2 for item, 0 for background

    Returns:
        tf.train.Example: tfrecord example contains all the info needed for training 
    """

    encoded_jpg = bayer_to_byte(bayer_path, image_dir)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classes_text = []
    label = []

    for anno in annos:
        label.append(anno["label"])
        xmin.append(anno["xmin"])
        xmax.append(anno["xmax"])
        ymin.append(anno["ymin"])
        ymax.append(anno["ymax"])


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

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def xywh2xyxy(bbox):
    "bbox is a list contains x,y,w,h"
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    return [xmin, ymin, xmax, ymax]


def xyxy2xywh(bbox):
    "bbox is a list contains xmin,ymin,xmax,ymax"
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) / 2
    h = (ymax - ymin) / 2
    return [x, y, w, h]


def read_yolo_anno(anno_path):
    """yolo anno [label, x, y, w, h] in normalized form 

    Args:
        anno_path (path): path to the yolo anno file

    Returns:
        dict: return a dict of yolo anno 
    """
    with open(anno_path, "r") as f:
        annos = []
        for line in f.readlines():
            data = line.strip().split()
            data = [float(x) for x in data]
            # SSD item label is 1
            anno = {}
            anno["label"] = int(data[0]) + 1
            x, y, w, h = data[1], data[2], data[3], data[4]
            [xmin, ymin, xmax, ymax] = xywh2xyxy([x, y, w, h])
            anno["xmin"] = xmin 
            anno["ymin"] = ymin 
            anno["xmax"] = xmax 
            anno["ymax"] = ymax 

            annos.append(anno)
    return annos


def read_xyxy_anno(anno_path):

    with open(anno_path, "r") as f:
        annos = []
        for line in f.readlines():
            data = line.strip().split()
            data = np.array(data, dtype=np.float32)
            # SSD item label is 1
            anno = {}
            anno["label"] = int(data[0]) + 1
            xmin, ymin, xmax, ymax = data[1], data[2], data[3], data[4]
            anno["xmin"] = xmin / 960
            anno["ymin"] = ymin / 540
            anno["xmax"] = xmax / 960 
            anno["ymax"] = ymax / 540

            annos.append(anno)
    return annos


def create_tfrecord(bayer_path, labels, image_dir, tfrecord_name):
    
    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for label in labels:
            basename = os.path.basename(label)
            bayer_basename = re.sub(".bayer_8.txt", ".bayer_8", basename)
            bayerfile_path = os.path.join(bayer_path, bayer_basename)
            annos = read_yolo_anno(label)
            example = create_single_example(bayerfile_path, annos, image_dir)
            writer.write(example.SerializeToString())



def create_sharded_tfrecord(bayer_path, label_path, image_dir, tfrecord_dir):
    labels = glob.glob(f"{label_path}/*.txt")
    print(len(labels))
    splits = int(len(labels) / 1000) + 1
    for i in range(splits):
        tfrecord_name = f"{tfrecord_dir}/tfrecord_{i:04d}.tfrecord"
        start = i * 1000
        end = min(len(labels), start + 1000)
        create_tfrecord(bayer_path , labels[start:end], image_dir, tfrecord_name)


if __name__ == "__main__":
    
    bayer_base_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/bayer"
    label_base_dir = re.sub("bayer", "labels", bayer_base_dir)
    stack_bayer_base_dir = re.sub("bayer", "stack_bayer", bayer_base_dir)
    tfrecord_base_dir = re.sub("bayer", "tfrecord", bayer_base_dir)

    folders = os.listdir(bayer_base_dir)
    folders = ["gen_grocery", "imagr_hands_faces_170322", "small_product"]

    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    for folder in folders:
        stack_bayer_dir = os.path.join(stack_bayer_base_dir, folder)
        tfrecord_dir = os.path.join(tfrecord_base_dir, folder)
        os.makedirs(stack_bayer_dir, exist_ok=True)
        os.makedirs(tfrecord_dir, exist_ok=True)
        bayer_path = os.path.join(bayer_base_dir, folder)
        label_path = os.path.join(label_base_dir, folder)

        pool.apply_async(create_sharded_tfrecord, args=(bayer_path, label_path, stack_bayer_dir, tfrecord_dir,))
    
    pool.close()
    pool.join()

