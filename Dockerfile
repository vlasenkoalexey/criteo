# Dockerfile

ARG BASE_IMAGE=''
FROM ${BASE_IMAGE}

# python is python3.7 and pip is pip3.7 in gcr.io/deeplearning-platform-release/tf2-gpu.2-1

WORKDIR /root

RUN pip install google-cloud-bigquery
RUN pip install google-cloud-bigquery-storage
RUN pip install google-cloud-logging


ENV PROJECT_ID=alekseyv-scalableai-dev
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/alekseyv-scalableai-dev-077efe757ef6.json

#ENV KMP_AFFINITY=""
#ENV TF_DISABLE_MKL=1
ENV OMP_NUM_THREADS=12
#RUN echo ${OMP_NUM_THREADS}

# RUN pip list

# Copies the trainer code
RUN mkdir /root/trainer
RUN mkdir /root/model
COPY alekseyv-scalableai-dev-077efe757ef6.json /root/
COPY trainer/trainer.py /root/trainer/trainer.py

# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "trainer/trainer.py", "/root/model"]