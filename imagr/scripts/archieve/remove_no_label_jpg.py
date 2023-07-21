import numpy as np
import os
import glob
import re

labels_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/labels/full_of_prods_dylan_desk"

labels = glob.glob(f"{labels_dir}/*/*.txt")
print(f"total images {len(labels)}")

counter = 0

for label in labels:
    img = re.sub(".txt", ".jpg", label)
    img = re.sub("/labels", "/images", img)

    if os.stat(label).st_size == 0:
        counter += 1
        os.remove(label)
        if os.path.exists(img): 
            os.remove(img)

print(f"empty frames {counter}")
print(f"value frames {len(labels) - counter}")
