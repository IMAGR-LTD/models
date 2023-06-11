MODEL_DIR="/mnt/saved_models/v2_p_det"
python3 object_detection/model_main.py \
    --logtostderr=true \
    --model_dir=$MODEL_DIR \
    --pipeline_config_path=$MODEL_DIR/pipeline.config
