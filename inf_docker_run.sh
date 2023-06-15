docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb \
-v $PWD:/home/tensorflow/models \
-v $PWD/data_imagr:/data \
-v $PWD/results:/results \
-w /home/tensorflow/models \
edgetpu_inf bash
