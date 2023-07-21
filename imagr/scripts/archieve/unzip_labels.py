import os 
import zipfile
import shutil


src_zipfile = "/home/walter/Downloads/task_ispd_images-2023_05_10_23_42_07-yolo 1.1.zip"
dst = "/home/walter/git/mobileDet/data/micro_controller/yolo_labels"
os.makedirs(dst, exist_ok=True)
zip_file = zipfile.ZipFile(src_zipfile, "r")
zip_file.extractall(dst)

obj_folder = os.path.join(dst, "obj_train_data")
if os.path.exists(obj_folder):
    for root, dirs, files in os.walk(obj_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                shutil.move(file_path, dst)
                print(file_path)


shutil.rmtree(obj_folder)
os.remove(os.path.join(dst, "obj.data"))
os.remove(os.path.join(dst, "obj.names"))
os.remove(os.path.join(dst, "train.txt"))
