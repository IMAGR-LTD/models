from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import glob 
import random 
import os

src_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/new_office/cam2"
save_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/od_blackout/cam2"
os.makedirs(save_dir, exist_ok=True)

imgs = glob.glob(f"{src_dir}/*.jpg")

# img = imgs[20]
# img = Image.open(img)
# img_np = np.array(img)
# plt.imshow(img_np)
# plt.show()


# imgs = random.sample(imgs, k=1000)

print(len(imgs))
for image in imgs:
    basename = os.path.basename(image)
    img = Image.open(image)
    img_np = np.array(img)

    x1, y1 = 324, 215
    x2, y2 = 0, 183
    x3, y3 = 0, 324  
    x4, y4 = 324, 324

    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=255)

    blacked_out_img = Image.composite(Image.new('RGB', img.size, (0, 0, 0)), img, mask)
    blacked_out_img.save(os.path.join(save_dir, basename))



'''
cam0 points
x1, y1 = 215, 244
x2, y2 = 125, 244
x3, y3 = 42, 324  
x4, y4 = 308, 324


cam1 points
x1, y1 = 324, 180
x2, y2 = 0, 214
x3, y3 = 0, 324  
x4, y4 = 324, 324


cam2 points
x1, y1 = 324, 215
x2, y2 = 0, 183
x3, y3 = 0, 324  
x4, y4 = 324, 324
'''