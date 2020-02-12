# Dockerfile
# TODO: replace with DLVM container after new version is released
# https://pantheon.corp.google.com/gcr/images/deeplearning-platform-release/GLOBAL/tf2-cpu
#FROM ubuntu:cosmic
#FROM tensorflow/tensorflow:2.0.0-gpu
#FROM tensorflow/tensorflow:2.1.0
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-1

# python is python3.7 and pip is pip3.7 in gcr.io/deeplearning-platform-release/tf2-gpu.2-1

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
	 python-pip python-dev libgomp1 curl && \
     rm -rf /var/lib/apt/lists/*

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

RUN pip install --upgrade pip

RUN pip install setuptools requests wheel
RUN pip install pandas numpy

RUN pip list

# Tensorflow
RUN pip install tensorflow==2.1.0
#RUN pip install tensorflow-gpu==2.0.0
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow-gpu==2.1.0
#RUN pip install tf-nightly-gpu==2.2.0.dev20200118
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow==2.1.0

# TF.IO
RUN pip install --no-deps tensorflow-io==0.11.0
# COPY dependencies/tensorflow_io-0.15.0-cp37-cp37m-manylinux2010_x86_64.whl /root/
# RUN pip install --no-deps /root/tensorflow_io-0.15.0-cp37-cp37m-manylinux2010_x86_64.whl

RUN pip install google-cloud-bigquery
RUN pip install google-cloud-bigquery-storage
RUN pip install google-cloud-logging

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
#RUN pip install cloudml-hypertune

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.1/extras/CUPTI/lib64/
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENV PROJECT_ID=alekseyv-scalableai-dev
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/alekseyv-scalableai-dev-077efe757ef6.json
#ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.1/extras/CUPTI/lib64/

RUN pip list

# Copies the trainer code
RUN mkdir /root/trainer
RUN mkdir /root/model
COPY alekseyv-scalableai-dev-077efe757ef6.json /root/
COPY trainer/trainer.py /root/trainer/trainer.py

# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "trainer/trainer.py", "/root/model"]