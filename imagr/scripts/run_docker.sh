sudo docker run -u $(id -u):$(id -g) -it --gpus device=3 -v "$(pwd)":/mnt  tf1_od  /bin/bash
