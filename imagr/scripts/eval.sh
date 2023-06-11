MODEL_DIR=/mnt/saved_models/anchor1
python3 object_detection/model_main.py \
  --model_dir=$MODEL_DIR \
  --pipeline_config_path=$MODEL_DIR/pipeline.config \
  --run_once \
  --checkpoint_dir=$MODEL_DIR
