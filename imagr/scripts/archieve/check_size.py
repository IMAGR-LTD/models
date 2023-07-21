import os 
import glob 

image_path = "/home/walter/nas_cv/walter_stuff/modular_dataset/ams_office/images/shop_20230117"
jpgs = glob.glob(f"{image_path}/*.jpg")

for jpg in jpgs:
    size = os.path.getsize(jpg)
    if size < 1500000:
        print(size)