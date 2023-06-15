MODEL="run1"
MODEL_DIR=/trained_model/$MODEL
python3 research/object_detection/model_main.py \
  --model_dir=$MODEL_DIR \
  --pipeline_config_path=$MODEL_DIR/pipeline.config \
  --run_once \
  --checkpoint_dir=$MODEL_DIR
