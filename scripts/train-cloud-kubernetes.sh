#set -v
DIR="$(cd "$(dirname "$0")" && pwd)"
source $DIR/train-common.sh

echo "Deleting existing resources..."
kubectl delete pod,service,PodSecurityPolicy --all
#kubectl delete pod,service criteo-sample-kubernetes-chief-0 criteo-sample-kubernetes-ps-0 criteo-sample-kubernetes-worker-0 criteo-sample-kubernetes-worker-1 --now

CLUSTER_NAME='criteo-cluster-tpu'
export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-central1-c"

# gsutil mb gs://${BUCKET_NAME}
#gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo


echo "Rebuilding docker image..."
echo "Docker base image: ${DOCKER_BASE_IMAGE}"
docker build -f Dockerfile -t $IMAGE_URI --build-arg BASE_IMAGE=${DOCKER_BASE_IMAGE} ./
docker push $IMAGE_URI

CURRENT_DATE_UTC=`date --utc -Iseconds`
python scripts/render_template.py $@ --job-dir=${MODEL_DIR} --image-name=${IMAGE_URI} | kubectl create -f -

echo "submitted job to kubernetes cluster"
echo ""
echo "log read command:"
echo "gcloud logging read 'resource.type=\"container\"
resource.labels.cluster_name=\"${CLUSTER_NAME}\"
resource.labels.namespace_id=\"default\"
resource.labels.project_id=\"${PROJECT_ID}\"
resource.labels.zone:\"${REGION}\"
resource.labels.container_name=\"tensorflow\"
timestamp>=\"${CURRENT_DATE_UTC}\"
' --limit 1000000000000 --order asc --format \"value(resource.labels.pod_id, jsonPayload.message, textPayload)\""
echo ""

export PROJECT_ID=alekseyv-scalableai-dev
while true; do
echo "logs since ${CURRENT_DATE_UTC}"
sleep 60
gcloud logging read "resource.type=\"container\"
resource.labels.cluster_name=\"${CLUSTER_NAME}\"
resource.labels.namespace_id=\"default\"
resource.labels.project_id=\"${PROJECT_ID}\"
resource.labels.zone:\"${REGION}\"
resource.labels.container_name=\"tensorflow\"
timestamp>=\"${CURRENT_DATE_UTC}\"
" --limit 1000000000000 --order asc --format 'value(resource.labels.pod_id, jsonPayload.message, textPayload)' > logfile.txt
cat logfile.txt | sed '/^$/d'
if [[ $(cat logfile.txt | head -n 5 | wc -l) -ne 0 ]]; then
    CURRENT_DATE_UTC=`date --utc -Iseconds`;
else
    kubectl get events --sort-by=.metadata.creationTimestamp | tail -n 2
fi

done
