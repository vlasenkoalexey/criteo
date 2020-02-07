#!/bin/bash
export PROJECT_ID=alekseyv-scalableai-dev
export GOOGLE_APPLICATION_CREDENTIALS="${PWD}/alekseyv-scalableai-dev-077efe757ef6.json"
export IMAGE_TAG=tf-nightly-dev20200118
export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

AI_PLATFROM_MODE=docker
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME=${CURRENT_DATE}

for i in "$@"
do
case $i in
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

MODEL_DIR="models/${MODEL_NAME}"
#mkdir -p ${MODEL_DIR}

if [ "$TENSORBOARD" = true ] ; then
    trap "kill 0" SIGINT
    echo "running tensorboard: tensorboard --logdir=${MODEL_DIR}/logs --port=0"
    tensorboard --logdir=${MODEL_DIR}/logs --port=0 &
fi

if [ "$AI_PLATFROM_MODE" = "docker" ] ; then
    echo "Rebuilding docker image..."
    docker build -f Dockerfile -t $IMAGE_URI ./
    docker push $IMAGE_URI

    echo "Running training job in docker image..."
    # Install NVIDIA driver and https://github.com/NVIDIA/nvidia-docker in order to be able to train models on GPU.
    docker run --gpus all -v ${PWD}/${MODEL_DIR}:/${MODEL_DIR} $IMAGE_URI python trainer/trainer.py --job-dir=/${MODEL_DIR} $@
    #docker run --gpus all -it -v ${PWD}:/host $IMAGE_URI python trainer/trainer.py --job-dir=/root/model $@
elif [ "$AI_PLATFROM_MODE" = "python" ] ; then
    echo "Running training job as a python script"
    python trainer/trainer.py --job-dir=${MODEL_DIR}
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