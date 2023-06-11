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
