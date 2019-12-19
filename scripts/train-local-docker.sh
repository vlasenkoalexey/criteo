#!/bin/bash
export PROJECT_ID=alekseyv-scalableai-dev
export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
export IMAGE_TAG=v1
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI
docker run $IMAGE_URI python trainer/trainer.py --job-dir=/root/model $@
#docker run -it -v ${PWD}:/host $IMAGE_URI python trainer/trainer.py --job-dir=/root/model $@
