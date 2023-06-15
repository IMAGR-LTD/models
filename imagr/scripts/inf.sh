MODEL_DIR="every_ten"
INPUT_DATASET="OD_instore_090623_testset"
python3 imagr/scripts/detect_image.py \
  --model /home/tensorflow/models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
  --labels /home/tensorflow/models/models_imagr/labels.txt \
  --input /data/images/$INPUT_DATASET \
  --output /results/$MODEL_DIR"_"$INPUT_DATASET