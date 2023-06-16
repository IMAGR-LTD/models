DATASET="OD_instore_090623_testset"
MODEL_DIR="4k_data"
python3 imagr/scripts/run_edgetpu_coco_metric.py \
-m /models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
-i /data/images/$DATASET \
-a /data/coco_gt/$DATASET/$DATASET.json \
-o /data/coco_dt/$DATASET