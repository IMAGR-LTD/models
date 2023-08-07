import os 
import re
# from bbox_format_conversion import cpwh_xyxy, tlwh_xyxy
from utils.bbox_format_conversion import cpwh_xyxy, tlwh_xyxy


def load_yolo_label(file_path, im_w=324, im_h=324):
    """yolo format is [cls_id, x_center, y_center, w, h] range from [0, 1]

    Args:
        file_path (path): path to label file end with .txt 

    return list of bbox [xmin, ymin, xmax, ymax] range from [0, 255]
    """
    assert os.path.exists(file_path)
    bboxs = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            record = line.strip().split()
            record = list(map(lambda x: float(x), record))
            bbox = record[1:]
            xmin, ymin, xmax, ymax = cpwh_xyxy(bbox)
            xmin = int(xmin * im_w)
            xmax = int(xmax * im_w)
            ymin = int(ymin * im_h)
            ymax = int(ymax * im_w)
            bbox = [xmin, ymin, xmax, ymax]
            bboxs.append(bbox)
    return bboxs


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


def save_label_file(bbox, dst, class_id=1):
    assert len(bbox) != 0
    
    parent_dir = os.path.dirname(dst)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(dst, 'w') as f: 
        line = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
        f.write(line)


def get_label_path_by(img_path):
    label_path = re.sub("/images", "/labels", img_path)
    label_path = re.sub(".jpg", ".txt", label_path)
    if not os.path.exists(label_path):
        print("label path not exist")
    return label_path

def get_img_path_by(label_path):
    img_path = re.sub("/labels", "/images", label_path)
    img_path = re.sub(".txt", ".jpg", img_path)
    if not os.path.exists(img_path):
        print("image path not exist")
    return img_path