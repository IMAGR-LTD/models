MODEL_DIR="4k_data"
INPUT_DATASET="OD_instore_090623_testset"
python3 imagr/scripts/run_edgetpu_inf_jpg.py \
  --model /models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
  --labels /models/models_imagr/labels.txt \
  --input /data/images/$INPUT_DATASET \
  --output /results/$MODEL_DIR"_"$INPUT_DATASET