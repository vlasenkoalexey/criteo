#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#set -v

export PROJECT_ID=alekseyv-scalableai-dev
export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-central1"

gsutil mb gs://${BUCKET_NAME}

#gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME=${CURRENT_DATE}
AI_PLATFROM_MODE='docker'

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
    --ai-platform-mode=*)
    AI_PLATFROM_MODE="${i#*=}"
    ;;
esac
done

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

JOB_NAME=train_${MODEL_NAME}
export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model

if [ "$TENSORBOARD" = true ] ; then
    trap "kill 0" SIGINT
    echo "running tensorboard: tensorboard --logdir=${MODEL_DIR}/logs --port=0"
    tensorboard --logdir=${MODEL_DIR}/logs --port=0 &
fi

if [ "$AI_PLATFROM_MODE" = "docker" ] ; then
    echo "Rebuilding docker image..."
    export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
    export IMAGE_TAG=v1
    export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
    docker build -f Dockerfile -t $IMAGE_URI ./
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
