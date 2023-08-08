import os
from PIL import Image, ImageDraw
import numpy as np
import glob
from collections import defaultdict
import pprint
import re
from utils.ios import load_yolo_label, save_label_file
import uuid


def xywh_xyxy(xywh):
    x, y, w, h = xywh
    xmin = x
    xmax = x + w
    ymin = y
    ymax = y + h
    return [xmin, ymin, xmax, ymax]


def read_label(file):
    with open(file, "r") as f:
        bboxs = []
        for line in f:
            obj = line.split()
            item, x, y, w, h = obj
            xmin, ymin, xmax, ymax = xywh_xyxy(
                map(lambda x: float(x), [x, y, w, h]))
            xmin, ymin, xmax, ymax = map(
                lambda x: int(x * 324), [xmin, ymin, xmax, ymax])
            bboxs.append([xmin, ymin, xmax, ymax])

    return bboxs


def stack_green_and_get_label(imgs_list, labels_list):
    """stack 3 images' green channel and form there bbox 

    Args:
        imgs_list (_type_): _description_
        labels_list (_type_): _description_

    Returns:
        PIL image
        bbox
    """
    assert len(imgs_list) == 3
    assert len(labels_list) == 3
    img1_np = np.array(Image.open(imgs_list[0]))
    img2_np = np.array(Image.open(imgs_list[1]))
    img3_np = np.array(Image.open(imgs_list[2]))
    stack_green_np = np.stack(
        [img1_np[:, :, 1], img2_np[:, :, 1], img3_np[:, :, 1]], axis=2)

    l1 = load_yolo_label(labels_list[0])
    l2 = load_yolo_label(labels_list[1])
    l3 = load_yolo_label(labels_list[2])
    xmin_1, ymin_1, xmax_1, ymax_1 = l1[0]
    xmin_2, ymin_2, xmax_2, ymax_2 = l2[0]
    xmin_3, ymin_3, xmax_3, ymax_3 = l3[0]
    xmin = min(xmin_1, xmin_2, xmin_3)
    ymin = min(ymin_1, ymin_2, ymin_3)
    xmax = max(xmax_1, xmax_2, xmax_3)
    ymax = max(ymax_1, ymax_2, ymax_3)

    # class id 1 means in, 2 means out
    class_id = 1
    if ymin_3 > ymin_1:
        class_id = 1
    else:
        class_id = 2

    return Image.fromarray(stack_green_np), [xmin, ymin, xmax, ymax], class_id


def split_by_cameras(labels):
    cam_dict_label_list = defaultdict(list)
    for label in labels:
        basename = os.path.basename(label)
        cam_id = basename.split("_")[1]
        cam_dict_label_list[cam_id].append(label)
    return cam_dict_label_list


def get_imgs_list_by(labels_list):
    imgs_list = []
    for label in labels_list:
        img_path = re.sub("/labels", "/images", label)
        img_path = re.sub(".txt", ".jpg", img_path)
        imgs_list.append(img_path)
    return imgs_list


def filter_valid_label(label):
    if not os.path.exists(label):
        return False
    else:
        with open(label, "r") as f:
            lines = len(f.readlines())
            return lines == 1


def get_img_and_label_list(labels_folder):
    labels = glob.glob(f"{labels_folder}/*.txt")
    labels = sorted(list(filter(filter_valid_label, labels)))
    imgs = get_imgs_list_by(labels)
    return imgs, labels


def run(imgs, labels, img_save, label_save, result_path):
    for i in range(len(imgs)-2):
        imgs_list = imgs[i:i+3]
        labels_list = labels[i:i+3]
        filename = uuid.uuid4()
        img, bbox, class_id = stack_green_and_get_label(imgs_list, labels_list)
        save_label_file(bbox, os.path.join(
            label_save, f"{filename}.txt"), class_id)
        img.save(os.path.join(img_save, f"{filename}.jpg"))
        ImageDraw.Draw(img).rectangle(bbox)
        img.save(os.path.join(result_path, f"{filename}.jpg"))


def get_consecutive_labels(labels):
    consecutive_frames = [[]]
    for i in range(len(labels)):
        frame = labels[i]
        basename = os.path.basename(frame)
        timestamp = int(basename.split('_')[0])
        if len(consecutive_frames[-1]) > 0:
            lastframe = consecutive_frames[-1][-1]
            last_timestamp = int(os.path.basename(lastframe).split("_")[0])
            if timestamp - last_timestamp < 60:
                consecutive_frames[-1].append(frame)
            else:
                consecutive_frames.append([frame])
        else:
            consecutive_frames[-1].append(frame)

    return consecutive_frames


cam = "cam_2"
img_save_dir = f"/home/walter/git/green_data/green/images/{cam}"
label_save_dir = f"/home/walter/git/green_data/green/labels/{cam}"
result_save_dir = f"/home/walter/git/green_data/green/results/{cam}"

label_raw_dir = f"/home/walter/git/green_data/labels/{cam}"
events = os.listdir(label_raw_dir)
for event in events:
    event_path = os.path.join(label_raw_dir, event)
    labels = list(sorted(glob.glob(f"{event_path}/*.txt")))

    consecutive_frames = get_consecutive_labels(labels)
    if len(consecutive_frames) >= 1:
        for consecutive_frame in consecutive_frames:
            if len(consecutive_frame) >= 3:
                imgs = get_imgs_list_by(consecutive_frame)
                # pprint.pprint(imgs)
                # pprint.pprint(consecutive_frame)
                # img_save = os.path.join(img_save_dir, cam, dataset, barcode)
                # label_save = os.path.join(label_save_dir, cam, dataset, barcode)
                # result_save = os.path.join(result_save_dir, cam, dataset, barcode)
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(label_save_dir, exist_ok=True)
                os.makedirs(result_save_dir, exist_ok=True)
                run(imgs, labels, img_save_dir, label_save_dir, result_save_dir)


# pprint.pprint(consecutive_frames)
# pprint.pprint(imgs)


# cams = os.listdir(label_raw_dir)
# for cam in cams:
#     label_dir = os.path.join(label_raw_dir, cam)
#     barcodes = os.listdir(label_dir)
#     for barcode in barcodes:
#         imgs, labels = get_img_and_label_list(os.path.join(label_raw_dir, cam, barcode))
#         img_save = os.path.join(img_save_dir, cam, dataset, barcode)
#         label_save = os.path.join(label_save_dir, cam, dataset, barcode)
#         result_save = os.path.join(result_save_dir, cam, dataset, barcode)
#         os.makedirs(img_save, exist_ok=True)
#         os.makedirs(label_save, exist_ok=True)
#         os.makedirs(result_save, exist_ok=True)
#         run(imgs, labels, img_save, label_save, result_save)
