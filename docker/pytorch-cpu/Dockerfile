ARG UBUNTU_VERSION=20.04

#ARG DVC_AWS_ACCESS_KEY_ID
#ARG DVC_AWS_SECRET_ACCESS_KEY

#ARG MLFLOW_TRACKING_URI
#ARG MLFLOW_S3_ENDPOINT_URL
#ARG AWS_ACCESS_KEY_ID
#ARG AWS_SECRET_ACCESS_KEY

#ARG GIT_REPO


FROM ubuntu:${UBUNTU_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

RUN python3 -m pip --no-cache-dir install --upgrade \
    setuptools 
    
RUN pip3 install --no-cache-dir grpcio==1.36.1

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN pip3 install --no-cache-dir torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

WORKDIR /app

ADD requirements.txt /
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

ENTRYPOINT ["/bin/sh"]