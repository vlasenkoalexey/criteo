# Dockerfile
# TODO: replace with DLVM container after new version is released
# https://pantheon.corp.google.com/gcr/images/deeplearning-platform-release/GLOBAL/tf2-cpu
#FROM ubuntu:cosmic
#FROM tensorflow/tensorflow:2.0.0-gpu
#FROM tensorflow/tensorflow:2.1.0
#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM tensorflow/tensorflow:nightly-gpu-py3
ARG BASE_IMAGE=''
FROM ${BASE_IMAGE}

# python is python3.7 and pip is pip3.7 in gcr.io/deeplearning-platform-release/tf2-gpu.2-1

WORKDIR /root

RUN pip list

# Tensorflow
#RUN pip install tensorflow==2.0.0
#RUN pip install tensorflow-gpu==2.0.0
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow-gpu==2.1.0
#RUN pip install tf-nightly-gpu==2.2.0.dev20200205
#RUN pip install tf-nightly-gpu==2.2.0.dev20200118
#RUN pip install tf-nightly-gpu
#RUN pip install tensorflow==2.1.0

# TF.IO
#RUN pip install --no-deps tensorflow-io==0.11.0
#COPY dependencies/tensorflow_io-0.15.0-cp36-cp36m-manylinux2010_x86_64.whl /root/
#RUN pip install --no-deps /root/tensorflow_io-0.15.0-cp36-cp36m-manylinux2010_x86_64.whl

RUN pip install google-cloud-bigquery
RUN pip install google-cloud-bigquery-storage
RUN pip install google-cloud-logging

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
#RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN curl -O \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

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