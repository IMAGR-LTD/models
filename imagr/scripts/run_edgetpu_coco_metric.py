from PIL import Image
import glob
import json
import argparse
import os 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_edgetpu_model_coco_result(model_path, images_dir, result_json_saveDir):
    # init edgetpu models 
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # read images 
    imgs = glob.glob(f"{images_dir}/*.jpg")
    imgs = sorted(imgs)
    detections = []
    img_id = 1
    for img in imgs:
        image = Image.open(img)
        _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        interpreter.invoke()
        objs = detect.get_objects(interpreter, 0.5, scale)

        for obj in objs:
            bbox = obj.bbox 
            xmin = bbox.xmin
            ymin = bbox.ymin
            xmax = bbox.xmax 
            ymax = bbox.ymax 
            w = xmax - xmin 
            h = ymax - ymin 
            bbox = [xmin, ymin, w, h]
            score = obj.score 
            id = 1
            detection = {
                "image_id" : img_id,
                "category_id": 1,
                "bbox": bbox,
                "score": score 
            }
            detections.append(detection)
        
        img_id += 1
    print(detections)
    
    result_json_savePath = os.path.join(result_json_saveDir, "results.json")
    with open(result_json_savePath, 'w') as f:
        json.dump(detections, f)
    
    return result_json_savePath


def eval_coco_result(coco_gt, coco_dt):
    cocoGt = COCO(coco_gt)
    imgIds = cocoGt.getImgIds()
    cocoDt = cocoGt.loadRes(coco_dt)

    annType = 'bbox'
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='model path')
    parser.add_argument('-i', '--images', required=True, help='images dir')
    parser.add_argument('-a', '--coco_gt', required=True, help='coco groud true anno path')
    parser.add_argument('-o', '--output', required=True, help='coco detect result save path')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    coco_dt = get_edgetpu_model_coco_result(args.model, args.images, args.output)
    eval_coco_result(args.coco_gt, coco_dt)


if __name__ == '__main__':  
    main()