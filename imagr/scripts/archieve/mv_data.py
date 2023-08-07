import os 
import glob 
import shutil
import pprint

cwd = "/home/walter/nas_cv/walter_stuff/stack_green_channel/raw/images/od_no_skip"
barcodes = [0, 1, 2, 3, 4, 5]

for barcode in barcodes:
    barcode_dir = os.path.join(cwd, str(barcode))
    files = glob.glob(f"{barcode_dir}/*.jpg")
    
    for file in files:
        basename = os.path.basename(file)
        cam_id = basename.split("_")[1]
        if cam_id == str(100):
            dst = os.path.join(cwd, "cam_0", str(barcode), basename)
            shutil.copy(file, dst)
        if cam_id == str(101):
            dst = os.path.join(cwd, "cam_1", str(barcode), basename)
            shutil.copy(file, dst)
        if cam_id == str(102):
            dst = os.path.join(cwd, "cam_2", str(barcode), basename)
            shutil.copy(file, dst)
            # pprint.pprint(file)
    # print(files)