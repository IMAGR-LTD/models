#!/bin/bash

set -ex

while getopts ":o:d:" opt; do
  case $opt in
    o) OUTPUT_DIR="$OPTARG"
    ;;
    d) DATA_DIR="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

if [ -z "$DATA_DIR" ]
then
    echo "-d can't be empty, need to provide data directory"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]
then
    echo "-o can't be empty, need to provide output directory"
    exit 1
fi


# Train
docker run --gpus device=0 -v $OUTPUT_DIR:/trained_model -v $DATA_DIR:/data \
australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:67a1146 \
python3 models/research/object_detection/model_main.py \
    --logtostderr=true \
    --model_dir=/trained_model \
    --pipeline_config_path=/trained_model/pipeline.config


# Write latest checkpoint name to a file
docker run -v $OUTPUT_DIR:/trained_model australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:67a1146 \
python3 models/imagr/scripts/get_last_ckpts.py /trained_model last_ckpts.txt

LAST_CHECKPOINT=$(cat $OUTPUT_DIR/last_ckpts.txt)

echo "FOUND LASTCHECKPOINT $LAST_CHECKPOINT"

mkdir -p /trained_model/export

# Export to SDD graph
docker run -v $OUTPUT_DIR:/trained_model australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:67a1146 \
python3 models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path=/trained_model/pipeline.config \
  --trained_checkpoint_prefix=$LAST_CHECKPOINT \
  --output_directory=/trained_model/export \
  --config_override " \
      model{ \
        ssd{ \
          post_processing { \
            batch_non_max_suppression { \
              score_threshold: 0.5 \
              iou_threshold: 0.2 \
            } \
          } \
        } \
      } \
  " \
  --add_postprocessing_op=true


# Export to tflite
docker run -v $OUTPUT_DIR:/trained_model tensorflow/tensorflow:2.11.0
tflite_convert \
--enable_v1_converter  \
--output_file="/trained_model/export/model.tflite"   \
--graph_def_file="/trained_model/export/tflite_graph.pb"   \
--inference_type=QUANTIZED_UINT8   \
--input_arrays="normalized_input_image_tensor"   \
--output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"   \
--mean_values=128   \
--std_dev_values=128   \
--input_shapes=1,320,320,3   \
--change_concat_input_ranges=false   \
--allow_nudging_weights_to_use_fast_gemm_kernel=true   \
--allow_custom_ops


# Convert to edgetpu
docker run -v $OUTPUT_DIR:/trained_model gcr.io/ml-shared-c-c41d/ml/edgetpu_compiler:7c837b2 \
edgetpu_compiler /trained_model/export/model.tflite -o /trained_model/export
