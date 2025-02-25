# models

This is IMAGR fork of https://github.com/tensorflow/models. There are imagr additions to train models.

## Object Detection

### Prepare the dataset folder 

Under `imagr_data`, you can place your dataset here. The object detection pipeline expects the dataset to be the following structure:

```
<dataset name>/
  label_map/
  tfrecord/
```

For example, if you want to import dataset from data-repository, you can

```
cd imagr_data
dvc import /mnt/nas_cv/data-repository labelled/20230601_cam0_microcontroller
```


### Prepare Pipeline config 

* Create a dir to save all the training output, checkpoint, export model, etc. 
  * eg. my models  path: `$PWD/models/imagr_models`
  * create a new folder `test_model_run` and put the `pipeline.config` in here
* update the config file, specificaly what tfrecord and label map to use

```
train_input_reader: {
  label_map_path: "/data/label_map/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "/data/tfrecord/set_1_loc_c.tfrecord"
  }
}
eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  num_examples: 8000
}
eval_input_reader: {
  label_map_path: "/data/label_map/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/data/tfrecord/set_1_loc_c.tfrecord"
  }
}
```

### Run the training pipeline

Run the pipeline script to start the training pipeline. This pipeline script will

1. Train the model
2. Export the latest model checkpoint to SSD graph
3. Convert the SSD graph to tflite file
4. Concvert the tflite to edgetpu file

```console
bash imagr/scripts/pipeline.sh -o $PWD/imagr_models/test_model_run -d $PWD/imagr_data/20230601_cam0_microcontroller
```


### Interactive shell

If you want to have an interactive shell then you can run the docker container and mount the whole folder

```bash
# Assuming you are in the models directory
docker run -it -v $PWD:/home/tensorflow/models australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:585776b bash
```