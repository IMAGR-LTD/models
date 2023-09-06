import numpy as np
import os
import glob
import re
import multiprocessing as mp



def remove(label):
    img = re.sub(".txt", ".jpg", label)
    img = re.sub("/labels", "/images", img)
    if os.stat(label).st_size == 0:
        os.remove(label)
        print("remove label")
        if os.path.exists(img): 
            os.remove(img)
            print("remove img")

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

labels_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/labels/new_office"

labels = glob.glob(f"{labels_dir}/*/*.txt")
print(f"total images {len(labels)}")

for label in labels:
    pool.apply_async(remove, args=(label,))
    #remove(label)
    
pool.close()
pool.join()

# print(f"empty frames {counter}")
# print(f"value frames {len(labels) - counter}")
