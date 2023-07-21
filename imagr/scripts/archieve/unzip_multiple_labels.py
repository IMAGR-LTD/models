import os 
import zipfile
import shutil
import glob
import pprint

img_savedir = "/home/walter/nas_cv/walter_stuff/crop_results/images"
labels_savedir = "/home/walter/nas_cv/walter_stuff/crop_results/labels"
tmp_dir = "/home/walter/nas_cv/walter_stuff/crop_results/tmp"
os.makedirs(img_savedir, exist_ok=True)
os.makedirs(labels_savedir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
zip_dir = "/home/walter/nas_cv/walter_stuff/crop_results/zipfiles"
zipfiles = glob.glob(f"{zip_dir}/*.zip")

for zip_file in zipfiles:
    barcode = zip_file.split("/")[-1].split("_")[1].split("-")[0]
    img_savepath = os.path.join(img_savedir, barcode)
    label_savepath = os.path.join(labels_savedir, barcode)
    os.makedirs(img_savepath, exist_ok=True)
    os.makedirs(label_savepath, exist_ok=True)
    
    zip_file = zipfile.ZipFile(zip_file, "r")
    zip_file.extractall(tmp_dir)

    obj_folder = os.path.join(tmp_dir, "obj_train_data")
    if os.path.exists(obj_folder):
        for root, dirs, files in os.walk(obj_folder):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    shutil.move(file_path, label_savepath)
                    print(file_path)
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    shutil.move(file_path, img_savepath)
                    print(file_path)


# shutil.rmtree(obj_folder)
# os.remove(os.path.join(dst, "obj.data"))
# os.remove(os.path.join(dst, "obj.names"))
# os.remove(os.path.join(dst, "train.txt"))