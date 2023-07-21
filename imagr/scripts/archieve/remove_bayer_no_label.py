import os 
import glob 
import shutil 
import re

bayer_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/bayer/imagr_hands_faces_170322/"
label_dir = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_multi_cls/labels/imagr_hands_faces_170322/"

bayers = glob.glob(f"{bayer_dir}/*.bayer_8")
labels = glob.glob(f"{label_dir}/*.txt")


print(len(bayers))
print(len(labels))

i = 0
for bayer in bayers:
    label = re.sub("/bayer", "/labels", bayer)
    label = re.sub(".bayer_8", ".bayer_8.txt", label)
    if label not in labels:
        print(bayer)
        i += 1
        os.remove(bayer)
print(i)