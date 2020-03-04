# Dockerfile
# TODO: replace with DLVM container after new version is released
# https://pantheon.corp.google.com/gcr/images/deeplearning-platform-release/GLOBAL/tf2-cpu
#FROM ubuntu:cosmic
#FROM tensorflow/tensorflow:2.0.0-gpu
#FROM tensorflow/tensorflow:2.1.0-gpu
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-1
FROM tensorflow/tensorflow:2.1.0-gpu-py3

# python is python3.7 and pip is pip3.7 in gcr.io/deeplearning-platform-release/tf2-gpu.2-1

# Installs necessary dependencies.
# RUN apt-get update && apt-get install -y --no-install-recommends \
# 	 python-pip python-dev libgomp1 curl && \
#      rm -rf /var/lib/apt/lists/*

WORKDIR /root

# RUN echo 'installing conda'
# RUN curl -O https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh
# RUN bash ./Anaconda2-2019.10-Linux-x86_64.sh -b
# RUN rm Anaconda2-2019.10-Linux-x86_64.sh
# ENV PATH /root/anaconda2/bin:$PATH
# RUN conda update conda
# RUN conda create -n tf2.0 python=2.7 tensorflow=2.0 -y
# RUN conda init bash
# RUN conda activate tf2.0
RUN python --version
RUN pip install --upgrade pip

RUN pip install setuptools requests wheel
RUN pip install pandas numpy

RUN pip list

# Tensorflow
#RUN pip install tensorflow==2.0.0
#RUN pip install tensorflow-gpu==2.0.0
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow-gpu==2.1.0
RUN pip install tf-nightly-gpu==2.2.0.dev20200303
#RUN pip install tf-nightly-gpu==2.2.0.dev20200227
#RUN pip install tf-nightly-gpu==2.2.0.dev20200201
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow==2.1.0

# TF.IO
#RUN pip install --no-deps tensorflow-io==0.11.0
COPY dependencies/tensorflow_io-2.2.0.dev20200227-cp36-cp36m-manylinux2010_x86_64.whl /root/
RUN pip install --no-deps /root/tensorflow_io-2.2.0.dev20200227-cp36-cp36m-manylinux2010_x86_64.whl

# COPY dependencies/tensorflow_io-2.2.0.dev20200227-cp37-cp37m-manylinux2010_x86_64.whl /root/
# RUN pip install --no-deps /root/tensorflow_io-2.2.0.dev20200227-cp37-cp37m-manylinux2010_x86_64.whl
#COPY dependencies/tensorflow_io-0.15.0-cp37-cp37m-manylinux2010_x86_64.whl /root/
#RUN pip install --no-deps /root/tensorflow_io-0.15.0-cp37-cp37m-manylinux2010_x86_64.whl

RUN pip install google-cloud-bigquery
RUN pip install google-cloud-bigquery-storage
RUN pip install google-cloud-logging
RUN pip install google-cloud-storage

RUN pip install tensorboardX


# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.1/extras/CUPTI/lib64/


ENV PROJECT_ID=alekseyv-scalableai-dev
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/alekseyv-scalableai-dev-077efe757ef6.json
#ENV TF_FORCE_GPU_ALLOW_GROWTH=true

RUN pip list

# Copies the trainer code
RUN mkdir /root/trainer
RUN mkdir /root/model
COPY alekseyv-scalableai-dev-077efe757ef6.json /root/
COPY trainer/trainer.py /root/trainer/trainer.py

# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "trainer/trainer.py", "/root/model"]
