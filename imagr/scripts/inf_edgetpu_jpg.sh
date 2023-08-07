MODEL_DIR="from_scratch"
INPUT_DATASET="rgb"
python3 imagr/scripts/run_edgetpu_inf_jpg_copy.py \
  --model /models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
  --labels /models/models_imagr/labels.txt \
  --input /data/images/$INPUT_DATASET \
  --output /results/$MODEL_DIR"_"$INPUT_DATASET