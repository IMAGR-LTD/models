import os
import io
import tensorflow as tf
import hashlib
import re
import glob
import numpy as np


def load_xyxy_label(file):
    """load xmin, ymin, xmax, ymax range from [0, 255] label
    return class ids and bboxs

    Args:
        file (path): path to the txt file 

    Returns:
        list, list: list of class ids and list of bbox
    """
    bboxs = []
    cls_ids = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.split()
            data = list(map(lambda x: float(x), data))
            cls_ids.append(data[0])
            xmin = data[1]
            ymin = data[2]
            xmax = data[3]
            ymax = data[4]
            bboxs.append([xmin, ymin, xmax, ymax])
    return cls_ids, bboxs 


def get_label_path_by(img_path):
    label_path = re.sub("/images", "/labels", img_path)
    label_path = re.sub(".jpg", ".txt", label_path)
    if not os.path.exists(label_path):
        print("label path not exist")
    return label_path


def generate_example(img_path, im_w=324, im_h=324, cls_id=1, cls_text=None):
    """
    cls_label is 1 for SSD Mobilenetv2 for item, 0 for background
    img_path: path to jpg image 
    bboxs: list of bbox [xmin, ymin, xmax, ymax] range from [0, 255]

    Returns:
        tf.train.Example: tfrecord example contains all the info needed for training 
    """
    assert os.path.exists(img_path)
    
    label_path = get_label_path_by(img_path)
    _, bboxs = load_xyxy_label(label_path)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_texts = []
    labels = []

    encoded_jpg = open(img_path, 'rb').read()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    for bbox in bboxs:
        labels.append(cls_id)
        xmins.append(float(bbox[0]/324))
        ymins.append(float(bbox[1]/324))
        xmaxs.append(float(bbox[2]/324))
        ymaxs.append(float(bbox[3]/324))
    
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_h])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_w])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_texts)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def create_tfrecord(imgs_path,  tfrecord_name):
    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for img_path in imgs_path:
            example = generate_example(img_path)
            if example:
                writer.write(example.SerializeToString())


def create_sharded_tfrecord(imgs,tfrecord_path):
    splits = int(len(imgs) / 1000) + 1
    for i in range(splits):
        tfrecord_name = f"{tfrecord_path}/{i:04d}.tfrecord"
        start = i * 1000
        end = min(len(imgs), start + 1000)
        create_tfrecord(imgs[start:end], tfrecord_name)



if __name__ == "__main__":
    img_dir = "/home/walter/nas_cv/walter_stuff/stack_green_channel/images/cam_0/od_no_skip" 
    tfrecord_dir = "/home/walter/nas_cv/walter_stuff/stack_green_channel/tfrecords/cam_0/od_no_skip"
    os.makedirs(tfrecord_dir, exist_ok=True)
    folders = os.listdir(img_dir)
    imgs_path = glob.glob(f"{img_dir}/*/*.jpg")
    print(len(imgs_path))
    create_sharded_tfrecord(imgs_path,tfrecord_dir)