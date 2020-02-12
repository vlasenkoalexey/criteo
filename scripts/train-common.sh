#!/bin/bash
export PROJECT_ID=alekseyv-scalableai-dev
export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
# TODO: replace this by TFE image once 2.1 is available
export DOCKER_CPU_BASE_IMAGE=gcr.io/deeplearning-platform-release/tf2-cpu.2-1
export DOCKER_GPU_BASE_IMAGE=gcr.io/deeplearning-platform-release/tf2-gpu.2-1

#export DOCKER_CPU_BASE_IMAGE=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#export DOCKER_GPU_BASE_IMAGE=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

#export IMAGE_TAG=tf21_gpu_tfofficial
#export IMAGE_TAG=tf21_gpu_cuda
#export IMAGE_TAG=tf21_gpu
export IMAGE_TAG=tf21_gpu
export DOCKER_BASE_IMAGE=${DOCKER_GPU_BASE_IMAGE}
export FLAVOR='GPU'

export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME=${CURRENT_DATE}
# MODEL_DIR_PREFIX can be specified by a calling script
if [ -n "${MODEL_DIR_PREFIX}" ]; then
    export MODEL_DIR=${MODEL_DIR_PREFIX}/${MODEL_NAME}/model
else
    export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model
fi
JOB_NAME=train_${MODEL_NAME}

for i in "$@"
do
case $i in
    --distribution-strategy=*)
    DISTRIBUTION_STRATEGY="${i#*=}"
    ;;
    --tensorboard)
    TENSORBOARD=true
    ;;
    --model-name=*)
    MODEL_NAME="${i#*=}"
    ;;
    --no-gpus=*)
    DOCKER_BASE_IMAGE=${DOCKER_CPU_BASE_IMAGE}
    export IMAGE_TAG=tf21_cpu
    export FLAVOR='CPU'
    ;;
    --ai-platform-mode=*)
    AI_PLATFROM_MODE="${i#*=}"
    ;;
esac
done

export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

if [ "$TENSORBOARD" = true ] ; then
    trap "kill 0" SIGINT
    echo "running tensorboard: tensorboard --logdir=${MODEL_DIR}/logs --port=0"
    tensorboard --logdir=${MODEL_DIR}/logs --port=0 &
fi

# one time operations
#gsutil mb gs://${BUCKET_NAME}
#gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo
