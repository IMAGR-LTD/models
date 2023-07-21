from PIL import Image
import re

img_path = "/home/walter/git/pipeline/models/data_imagr/images/od_skip_0/0/0_100_00_0000.jpg"
label_path = re.sub("/images", "labels", img_path)
label_path = re.sub("/.jpg", ".txt", label_path)
