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

# This is the common setup.
echo "Submitting an AI Platform job..."

TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1

export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-central1"
export MODEL_NAME="criteo_kaggle_gpu" # change to your model name
export PYTHON_VERSION="3.7"

PACKAGE_PATH=./trainer # this can be a gcs location to a zipped and uploaded package
export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model

gsutil mb gs://${BUCKET_NAME}

gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}

gcloud ai-platform jobs submit training ${JOB_NAME} \
        --package-path=${PACKAGE_PATH} \
        --module-name=trainer.trainer \
        --job-dir=${MODEL_DIR} \
        --region=us-central1 \
        --scale-tier=custom \
        --master-machine-type=n1-highcpu-16 \
        --master-accelerator=count=2,type=nvidia-tesla-k80 \
        --worker-count=2 \
        --worker-machine-type=n1-highcpu-16 \
        --worker-accelerator=count=2,type=nvidia-tesla-k80 \
        --stream-logs \
        -- \
	    ${MODEL_DIR}
set -

#         --parameter-server-count=3 \
#        --parameter-server-machine-type=n1-highmem-8 \

#--python-version=${PYTHON_VERSION} \
# Notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training
