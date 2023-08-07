import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import tensorflow as tf

import glob
import os 
import numpy as np


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def inf_one_image(image_path, interpreter, args, labels):
  # image = Image.open(image_path)
  image = np.fromfile(image_path,dtype=np.uint8)
  r = image[0::3].reshape((324,324))
  g = image[1::3].reshape((324,324))
  b = image[2::3].reshape((324,324))
  image = np.stack([r,g,b], axis=2)
  image = tf.image.resize(image, (320,320), method='nearest').numpy()
  image = Image.fromarray(image)
  _, scale = common.set_resized_input(
      interpreter, image.size, image)

  print('----INFERENCE TIME----')
  
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  objs = detect.get_objects(interpreter, args.threshold, scale)
  print('%.2f ms' % (inference_time * 1000))

  print('-------RESULTS--------')
  if not objs:
    print('No objects detected')

  for obj in objs:
    print(labels.get(obj.id, obj.id))
    print('  id:    ', obj.id)
    print('  score: ', obj.score)
    print('  bbox:  ', obj.bbox)

  if args.output:
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image_basename = os.path.basename(image_path)
    save_path = os.path.join(args.output, image_basename)
    image.save(save_path)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True,
                      help='input dir')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.5,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='output dir')
  args = parser.parse_args()
  os.makedirs(args.output, exist_ok=True)

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  all_images = glob.glob(f"{args.input}/*.rgb")
  for image_path in all_images:
    inf_one_image(image_path, interpreter, args, labels)

  


if __name__ == '__main__':
  main()
