ModelDir="/mnt/saved_models/v0_p_det"
LastCheckpoint="/model.ckpt-50000"
PipelineConfig=$ModelDir"/pipeline.config"
OutputDir=$ModelDir"/export"
python3 object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path=$PipelineConfig \
  --trained_checkpoint_prefix=$ModelDir$LastCheckpoint \
  --output_directory=$OutputDir \
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

tflite_convert \
  --output_file="$OutputDir/model.tflite" \
  --graph_def_file="$OutputDir/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="normalized_input_image_tensor" \
  --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,320,320,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops
