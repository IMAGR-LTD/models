from PIL import Image  
import glob 
import os 
import re
import numpy as np

src_dir = "/home/walter/git/pipeline/models/data_imagr/images/od_skip_0"
crop_save_dir = "/home/walter/git/Siamese_object_recognition/data"
labels_dir = re.sub("/images", "/labels", src_dir)

barcodes = os.listdir(labels_dir)
for barcode in barcodes:
    labels_path = os.path.join(labels_dir, barcode)
    crop_path = os.path.join(crop_save_dir, barcode)
    os.makedirs(crop_path, exist_ok=True)
    labels = glob.glob(f"{labels_path}/*.txt")
    for label in labels:
        img_path = re.sub("labels", "images", label)
        img_path = re.sub(".txt", ".jpg", img_path)
        basename = os.path.basename(img_path)
        
        with open(label, 'r') as f:
            anno = f.readline()
            bbox = anno.split()[1:]
            bbox = np.array(bbox, dtype=float)
            if bbox.any():
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                xmin = x 
                xmax = x + w
                ymin = y 
                ymax = y + h
                xmin = int(xmin * 324)
                xmax = int(xmax * 324)
                ymin = int(ymin * 324)
                ymax = int(ymax * 324)
                
                img = Image.open(img_path)
                crop = img.crop((xmin, ymin, xmax, ymax))
                crop.save(os.path.join(crop_path, basename))
                
        