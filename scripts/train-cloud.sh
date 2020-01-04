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

set -v

export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-central1"
export PYTHON_VERSION="3.7"
export RUNTIME_VERSION="1.14"

PACKAGE_PATH=./trainer # this can be a gcs location to a zipped and uploaded package
export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model

gsutil mb gs://${BUCKET_NAME}

gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME=${CURRENT_DATE}

for i in "$@"
do
case $i in
    --distribution-strategy=*)
    DISTRIBUTION_STRATEGY="${i#*=}"
    ;;
    --model-name=*)
    MODEL_NAME="${i#*=}"
            # unknown option
    ;;
esac
done

echo DISTRIBUTION_STRATEGY = ${DISTRIBUTION_STRATEGY}
echo 'DISTRIBUTION_STRATEGY:'
echo ${DISTRIBUTION_STRATEGY}

echo MODEL_NAME=${MODEL_NAME}
echo 'MODEL_NAME:'
echo ${MODEL_NAME}

case "${DISTRIBUTION_STRATEGY}" in
  "tf.distribute.MirroredStrategy" | "tf.distribute.experimental.CentralStorageStrategy")
   CONFIG="--master-accelerator=count=2,type=nvidia-tesla-k80"
   ;;
  "tf.distribute.experimental.ParameterServerStrategy")
   CONFIG="--parameter-server-count=1 --parameter-server-image-uri=${IMAGE_URI} --parameter-server-machine-type=n1-highcpu-16 --worker-count=2 --worker-machine-type=n1-highcpu-16 --worker-image-uri=${IMAGE_URI}"
   ;;
  "tf.distribute.experimental.MultiWorkerMirroredStrategy")
   CONFIG="--worker-image-uri=${IMAGE_URI} --worker-machine-type=n1-highcpu-16 --worker-count=2 --worker-accelerator=count=2,type=nvidia-tesla-k80"
   ;;
  *)
    echo "Invalid option ${DISTRIBUTION_STRATEGY}"
    ;;
esac

echo CONFIG = ${CONFIG}
echo 'CONFIG:'
echo ${CONFIG}

JOB_NAME=train_${MODEL_NAME}
export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model

echo "Submitting an AI Platform job..."
# see https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training
gcloud ai-platform jobs submit training ${JOB_NAME} \
        --config=${PWD}/scripts/config_fix.yaml \
        --scale-tier=CUSTOM \
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

