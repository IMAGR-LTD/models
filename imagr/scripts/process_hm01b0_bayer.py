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

file_path= '/home/walter/big_daddy/nigel/hm01b0_data/od_office_200723_blue'
save_dir = "/home/walter/nas_cv/walter_stuff/test/images/od_office_200723"
images=glob.glob(f"{file_path}/*.bayer")

for fname in images:
  fd=open(fname,'rb')
  bayer_8bit=np.fromfile(fd,dtype=np.uint8)
  bayer_8bit=bayer_8bit.reshape(324,324)
  linear_img=isp_it(bayer_8bit)
  img_gammad=np.array(255*(linear_img/255)**0.65,dtype='uint8')

  # # cv2.imshow('rgb image isp non linear',bayer_8bit)
  # # cv2.waitKey(5)

  savename = re.sub(".bayer", ".jpg", fname)
  savename = re.sub(file_path, save_dir, savename)
  os.makedirs(os.path.dirname(savename), exist_ok=True)
  
  # print(savename)
  basename = os.path.basename(savename)
  cam = basename.split("_")[4]
  if cam == "101"or cam == "102":
    rotated_image = cv2.rotate(img_gammad, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(savename,rotated_image,[cv2.IMWRITE_JPEG_QUALITY,100])
  else:
    rotated_image = cv2.rotate(img_gammad, cv2.ROTATE_180)
    cv2.imwrite(savename,rotated_image,[cv2.IMWRITE_JPEG_QUALITY,100])