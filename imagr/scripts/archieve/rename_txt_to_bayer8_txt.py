import os
import glob
import numpy
import re


label_path = "/home/walter/nas_cv/walter_stuff/modular_dataset/auk_office_test/labels/test"
labels = glob.glob(f"{label_path}/*[!_][0-9].txt")

print(len(labels))


for label in labels:
    old_name = label
    new_name = re.sub(".txt", ".bayer_8.txt", old_name)
    os.rename(old_name, new_name)
