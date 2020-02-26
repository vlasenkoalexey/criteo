#!/bin/bash


export MODEL_DIR_PREFIX="models/"
AI_PLATFROM_MODE=docker

DIR="$(cd "$(dirname "$0")" && pwd)"
source $DIR/train-common.sh

if [ "$AI_PLATFROM_MODE" = "docker" ] ; then
    echo "Rebuilding docker image..."
    echo "docker build -f Dockerfile -t $IMAGE_URI --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} ./"
    docker build -f Dockerfile -t $IMAGE_URI --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} ./
    docker push $IMAGE_URI

    echo "Running training job in docker image..."
    # Install NVIDIA driver and https://github.com/NVIDIA/nvidia-docker in order to be able to train models on GPU.
    # disable MKL if run locally, see https://b.corp.google.com/issues/149489290
    echo "docker run --gpus all -v ${PWD}/${MODEL_DIR}:/${MODEL_DIR} $IMAGE_URI python trainer/trainer.py --job-dir=/${MODEL_DIR} $@"
    docker run --gpus all -v ${PWD}/${MODEL_DIR}:/${MODEL_DIR} $IMAGE_URI python trainer/trainer.py --job-dir=/${MODEL_DIR} $@
    #docker run --gpus all -it -v ${PWD}:/host $IMAGE_URI python trainer/trainer.py --job-dir=/root/model $@
elif [ "$AI_PLATFROM_MODE" = "python" ] ; then
    echo "Running training job as a python script"
    #LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/google/home/alekseyv/cuda/extras/CUPTI/lib64/:/usr/local/google/home/alekseyv/cuda/targets/x86_64-linux/lib/; python3 trainer/trainer.py --job-dir=${MODEL_DIR} $@
    python3 trainer/trainer.py --job-dir=${MODEL_DIR} $@
else
    echo "Running training job using local Cloud AI command..."
    PACKAGE_PATH=./trainer
    gcloud ai-platform local train \
            --job-dir=${MODEL_DIR} \
            --module-name=trainer.trainer \
            --package-path=${PACKAGE_PATH} \
            -- \
            --job-dir=${MODEL_DIR} \
            $@
fi