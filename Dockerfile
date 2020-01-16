# Dockerfile
# TODO: replace with DLVM container after new version is released
# https://pantheon.corp.google.com/gcr/images/deeplearning-platform-release/GLOBAL/tf2-cpu
#FROM ubuntu:cosmic
#FROM tensorflow/tensorflow:2.0.0-gpu
#FROM tensorflow/tensorflow:2.1.0rc2-gpu
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

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
RUN pip install pandas
#RUN pip install tensorflow==2.0.0
#RUN pip install tensorflow-gpu==2.0.0
#COPY dependencies/tensorflow_gpu-2.1.0rc0-cp27-cp27mu-linux_x86_64.whl /root/
#RUN pip install /root/tensorflow_gpu-2.1.0rc0-cp27-cp27mu-linux_x86_64.whl
#RUN pip install tf-nightly-gpu
RUN pip install tensorflow-gpu==2.1.0
#RUN pip install tensorflow==2.1.0
RUN pip install --no-deps tensorflow-io==0.11.0
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

# Copies the trainer code
RUN mkdir /root/trainer
RUN mkdir /root/model
COPY alekseyv-scalableai-dev-077efe757ef6.json /root/
COPY trainer/trainer.py /root/trainer/trainer.py

# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "trainer/trainer.py", "/root/model"]