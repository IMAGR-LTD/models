from PIL import Image, ImageDraw
import glob
import json
import argparse
import os 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

def draw_objects(draw, objs):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],outline='red')
        # draw.text((bbox.xmin + 10, bbox.ymin + 10),'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),fill='red')

model_path = "/home/walter/git/pipeline/models/models_imagr/full_of_prods_dylan_desk/export/full_of_prods_dylan_desk.tflite"
# images_dir = "/home/walter/nas_cv/walter_stuff/test/images/od_office_200723"
images_dir = "/home/walter/git/pipeline/models/data_imagr/images/FP_OD_DYLAN_DESK"
# FP_data = "/home/walter/git/pipeline/models/data_imagr/images/FP_OD_DYLAN_DESK"

interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
imgs = glob.glob(f"{images_dir}/*.jpg")
imgs = sorted(imgs)

for img in imgs:
    image = Image.open(img)
    _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    objs = detect.get_objects(interpreter, 0.5, scale)
    print(len(objs))
    if len(objs) > 1:
        # basename = os.path.basename(img)
        # savepath = os.path.join(FP_data, basename)
        # image.save(savepath)
        draw_objects(ImageDraw.Draw(image), objs)
        image.show()


