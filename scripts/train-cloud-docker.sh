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

gcloud config configurations activate alekseyv-scalableai-dev

echo "Rebuilding docker image..."
export PROJECT_ID=alekseyv-scalableai-dev
export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
export IMAGE_TAG=v1
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI

# This is the common setup.
echo "Submitting an AI Platform job..."

TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1

export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-central1"
export MODEL_NAME="criteo_kaggle_docker_estimator" # change to your model name

PACKAGE_PATH=./trainer # this can be a gcs location to a zipped and uploaded package
export MODEL_DIR=gs://${BUCKET_NAME}/${MODEL_NAME}/model

gsutil mb gs://${BUCKET_NAME}

#gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${CURRENT_DATE}
IMAGE_URI=gcr.io/alekseyv-scalableai-dev/alekseyv_criteo_custom_container:v1

gcloud ai-platform jobs submit training ${JOB_NAME} \
        --config=${PWD}/scripts/config_fix.yaml \
        --region=${REGION} \
        --master-image-uri ${IMAGE_URI} \
        --worker-image-uri ${IMAGE_URI} \
        --stream-logs \
        -- python trainer/trainer.py ${MODEL_DIR}


set -

#--python-version=${PYTHON_VERSION} \
# Notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training
