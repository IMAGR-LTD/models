# Models

This is IMAGR fork of https://github.com/tensorflow/models. There are imagr additions to train models.

# Authenticate 

```bash 
# gcloud 
gcloud auth login 
gcloud config set project ml-shared-c-c41d
# docker 
gcloud auth configure-docker australia-southeast1-docker.pkg.dev
gcloud auth configure-docker gcr.io
    
```

# Track data and models 

```bash 
# make sure mount nas on /mnt/nas_cv
# keep track of data_imagr and models_imagr 
# everytime when you add new data to the data folder or train a model use 
dvc add models_imagr 
dvc add data_imagr 
git add models_imagr.dvc
git add data_imagr.dvc
dvc push 
# in the new location use 
dvc pull 
```



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

change the pipeline 

```bash
# under the models dir run 
bash imagr/scripts/pipeline.sh
```

# Eval checkpoint 

### Interactive shell

If you want to have an interactive shell then you can run the docker container and mount the whole folder

```bash 
# change the gpu id you want to use device=0
bash run_od_tf1_docker.sh
```



```bash
# Assuming you are in the models directory
docker run --gpus device=3 -it -v $PWD:/home/tensorflow/models \
-v $PWD/data_imagr:/data \
-v $PWD/models_imagr:/trained_model \
-w /home/tensorflow/models/research australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:585776b  bash
```



# Inference edgetpu on jpg  

```bash
# build docker container for inference 
docker build -f imagr/dockerfile/inf_edgetpu/Dockerfile -t edgetpu_inf .
bash run_edgetpu_docker.sh

# Change dataset and model
# MODEL_DIR="4k_data"
# INPUT_DATASET="OD_instore_090623_testset"
bash imagr/scripts/inf_edgetpu_jpg.sh
```

`run_edgetpu_docker.sh`

```bash
# --privileged flag is used to enable access to USB devices
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb \
-v $PWD:/models \
-v $PWD/data_imagr:/data \
-v $PWD/results:/results \
-w /models \
edgetpu_inf bash 
```

`inf_edgetpu_jpg.sh`

```bash
MODEL_DIR="4k_data"
INPUT_DATASET="OD_instore_090623_testset"
python3 imagr/scripts/run_edgetpu_inf_jpg.py \
  --model /models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
  --labels /models/models_imagr/labels.txt \
  --input /data/images/$INPUT_DATASET \
  --output /results/$MODEL_DIR"_"$INPUT_DATASET
```

# Eval edgetpu model with coco 

### Prepare coco ground true 

data_imagr

├── coco_dt
│   └── OD_instore_090623_testset
├── coco_gt
│   └── OD_instore_090623_testset
├── images
│   ├── OD_instore_090623_testset
├── label_map
└── tfrecord

```bash 
# name the ground true coco file with the same name as the dataset name 
# eg. OD_instore_090623_testset.json under 
# coco_gt-> OD_instore_090623_testset -> OD_instore_090623_testset.json
```

### Update the model and the testset 

```bash 
DATASET="OD_instore_090623_testset"
MODEL_DIR="4k_data"
python3 imagr/scripts/run_edgetpu_coco_metric.py \
-m /models/models_imagr/$MODEL_DIR/export/model_edgetpu.tflite \
-i /data/images/$DATASET \
-a /data/coco_gt/$DATASET/$DATASET.json \
-o /data/coco_dt/$DATASET
```

### Start the docker and Run the script 

```bash 
bash run_edgetpu_docker.sh
# inside the docker run 
bash imagr/scripts/eval_edgetpu_coco.sh
```

