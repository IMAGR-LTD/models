FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip wget curl 

# Install EdgeTPU runtime
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && apt-get install -y libedgetpu1-std

RUN apt-get install -y python3-pycoral

# Install Python dependencies
RUN pip3 install numpy pillow pycocotools
