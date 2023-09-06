import os 
import glob 
import re

labels_path = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/labels/new_office_onboard"
labels = glob.glob(f"{labels_path}/*/*/*.txt")
print(len(labels))

def get_img_path_by(label_path):
    img_path = re.sub("/labels", "/images", label_path)
    img_path = re.sub(".txt", ".jpg", img_path)
    if not os.path.exists(img_path):
        print("image path not exist")
    return img_path


for label in labels:
    if os.path.getsize(label) == 0:
        jpg_path = get_img_path_by(label)
        os.remove(label)
        os.remove(jpg_path)
