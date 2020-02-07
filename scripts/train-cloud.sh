#!/bin/bash
#set -v

export REGION="us-central1"
AI_PLATFROM_MODE='docker'

DIR="$(cd "$(dirname "$0")" && pwd)"
source $DIR/train-common.sh

echo DISTRIBUTION_STRATEGY = ${DISTRIBUTION_STRATEGY}
echo 'DISTRIBUTION_STRATEGY:'
echo ${DISTRIBUTION_STRATEGY}

echo MODEL_NAME=${MODEL_NAME}
echo 'MODEL_NAME:'
echo ${MODEL_NAME}

SCALE_TIER="CUSTOM"
CONFIG_FIX="--config=${PWD}/scripts/config_fix.yaml"
CONFIG="--master-machine-type=n1-highcpu-16 --master-accelerator=count=2,type=nvidia-tesla-k80"

case "${DISTRIBUTION_STRATEGY}" in
  "tf.distribute.MirroredStrategy")
   CONFIG="--master-machine-type=n1-highcpu-16 --master-accelerator=count=2,type=nvidia-tesla-k80"
   ;;
  "tf.distribute.experimental.ParameterServerStrategy")
   CONFIG="--master-machine-type=n1-highcpu-16 --parameter-server-count=1 --parameter-server-image-uri=${IMAGE_URI} --parameter-server-machine-type=n1-highcpu-16 --worker-count=2 --worker-machine-type=n1-highcpu-16 --worker-image-uri=${IMAGE_URI}"
   ;;
  "tf.distribute.experimental.CentralStorageStrategy")
   # See https://b.corp.google.com/issues/148108526 why PS is required
   CONFIG="--master-machine-type=n1-highcpu-16 --parameter-server-count=1 --parameter-server-image-uri=${IMAGE_URI} --parameter-server-machine-type=n1-highcpu-16"
   ;;
  "tf.distribute.experimental.MultiWorkerMirroredStrategy")
   CONFIG="--master-machine-type=n1-highcpu-16 --master-accelerator=count=2,type=nvidia-tesla-k80 --worker-image-uri=${IMAGE_URI} --worker-machine-type=n1-highcpu-16 --worker-count=2 --worker-accelerator=count=2,type=nvidia-tesla-k80"
   ;;
  "tf.distribute.experimental.TPUStrategy")
   CONFIG="--tpu-tf-version=1.14"
   SCALE_TIER="BASIC_TPU"
   ;;
  *)
    # If distribution strategy is not set, don't replace 'master' -> 'chief',
    # otherwise system will assume that environment works in distributed setting and
    # will expect to be executed in distribution strategy scope.
    # See https://github.com/tensorflow/tensorflow/blob/64c3d382cadf7bbe8e7e99884bede8284ff67f56/tensorflow/python/distribute/multi_worker_util.py#L235
    # Fixed in TF2.1rc2 https://github.com/tensorflow/tensorflow/commit/0390084145761a1d4da3be2bec8c56a28399db14
    CONFIG_FIX=""
    echo "Invalid option ${DISTRIBUTION_STRATEGY}"
    ;;
esac

echo CONFIG = ${CONFIG}
echo 'CONFIG:'
echo ${CONFIG}

if [ "$AI_PLATFROM_MODE" = "docker" ] ; then
    echo "Rebuilding docker image..."
    docker build -f Dockerfile -t $IMAGE_URI --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} ./
    docker push $IMAGE_URI

    echo "Submitting an AI Platform job..."
    # see https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training
    gcloud beta ai-platform jobs submit training ${JOB_NAME} \
            ${CONFIG_FIX} \
            --scale-tier=${SCALE_TIER} \
            --region=${REGION} \
            --master-image-uri=${IMAGE_URI} \
            --stream-logs \
            ${CONFIG} \
            -- python trainer/trainer.py --job-dir=${MODEL_DIR} --train-location=cloud $@
else
    export PYTHON_VERSION="3.7"
    export RUNTIME_VERSION="1.14"
    PACKAGE_PATH=./trainer # this can be a gcs location to a zipped and uploaded package

    echo "Submitting an AI Platform job..."
    # see https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training
    gcloud ai-platform jobs submit training ${JOB_NAME} \
            --config=${PWD}/scripts/config_fix.yaml \
            --scale-tier=${SCALE_TIER} \
            --job-dir=${MODEL_DIR} \
            --runtime-version=${RUNTIME_VERSION} \
            --region=${REGION} \
            --module-name=trainer.trainer \
            --package-path=${PACKAGE_PATH}  \
            --master-machine-type=n1-highcpu-16 \
            --stream-logs \
            ${CONFIG} \
            -- \
            --job-dir=${MODEL_DIR} --train-location=cloud $@
fi
