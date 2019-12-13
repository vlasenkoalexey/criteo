# Dockerfile
FROM ubuntu:cosmic

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
	 python-pip python-dev libgomp1 curl && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN pip install --upgrade pip
RUN pip install setuptools requests wheel
RUN pip install pandas
RUN pip install tensorflow==2.0.0
RUN pip install tensorflow-io==0.10.0
RUN pip install google-cloud-bigquery
RUN pip install google-cloud-bigquery-storage
RUN pip install google-cloud-logging

#RUN pip install tf-nightly==2.1.0.dev20191126


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

#COPY dependencies/tensorflow_io-0.10.0-cp27-cp27mu-manylinux2010_x86_64.whl /root
#RUN pip install --no-deps /root/tensorflow_io-0.10.0-cp27-cp27mu-manylinux2010_x86_64.whl

# Copies the trainer code
RUN mkdir /root/trainer
RUN mkdir /root/model
COPY alekseyv-scalableai-dev-077efe757ef6.json /root/
COPY trainer/trainer.py /root/trainer/trainer.py


# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "trainer/trainer.py", "/root/model"]