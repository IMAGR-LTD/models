# process raw bayer hm01b0 files

import numpy as np
import cv2
import os
import glob
import re


def isp_it(img):

  bayer_norm = (img/255).astype(np.float32)

  #red channel
  bayer_norm[1::2,1::2]=bayer_norm[1::2,1::2]
  #blue channel
  bayer_norm[::2,::2]=bayer_norm[::2,::2]*1.25
  bayer_norm=bayer_norm*3
  

  bayer_norm= np.clip(bayer_norm,0,1)
  bayer=np.array(bayer_norm*255,dtype=np.uint8)
  img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)  #was BG2RGB  which equates to the same thing but is wrong

  return img_rgb

for cam in ["cam_0", "cam_1", "cam_2"]:


  file_path= f'/home/walter/big_daddy/nigel/hm01b0_data/event_Data_020823_2/{cam}/something'
  save_dir = f"/home/walter/git/pipeline/models/data_imagr/images/{cam}"
  images=glob.glob(f"{file_path}/*/*.bayer")

  for fname in images:
    fd=open(fname,'rb')
    bayer_8bit=np.fromfile(fd,dtype=np.uint8)
    bayer_8bit=bayer_8bit.reshape(324,324)
    linear_img=isp_it(bayer_8bit)
    img_gammad=np.array(255*(linear_img/255)**0.65,dtype='uint8')

    # # cv2.imshow('rgb image isp non linear',bayer_8bit)
    # # cv2.waitKey(5)
    
    parent_dir = os.path.dirname(fname)
    parent_dir_name = os.path.basename(parent_dir)

    basename = os.path.basename(fname)
    savename = re.sub(".bayer", ".jpg", basename)
    # savename = re.sub(file_path, save_dir, savename)
    dir = os.path.join(save_dir, parent_dir_name)
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, savename)

    print(save_path)
    
    cam = basename.split("_")[2]
    if cam == "1"or cam == "2":
      rotated_image = cv2.rotate(img_gammad, cv2.ROTATE_90_COUNTERCLOCKWISE)
      cv2.imwrite(save_path,rotated_image,[cv2.IMWRITE_JPEG_QUALITY,100])
    else:
      rotated_image = cv2.rotate(img_gammad, cv2.ROTATE_180)
      cv2.imwrite(save_path,rotated_image,[cv2.IMWRITE_JPEG_QUALITY,100])