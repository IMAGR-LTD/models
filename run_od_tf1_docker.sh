docker run --gpus device=3 -it -v $PWD:/home/tensorflow/models \
-v $PWD/data_imagr:/data \
-v $PWD/models_imagr:/trained_model \
-w /home/tensorflow/models australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/object_detection_tf1:585776b  bash
