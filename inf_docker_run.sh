docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb \
-v $PWD:/home/tensorflow/models \
-w /home/tensorflow/models \
edgetpu_inf bash
