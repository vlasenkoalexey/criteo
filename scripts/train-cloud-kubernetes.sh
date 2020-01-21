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

python scripts/render_template.py $@ | kubectl delete -f -

echo "Rebuilding docker image..."
export PROJECT_ID=alekseyv-scalableai-dev
export IMAGE_REPO_NAME=alekseyv_criteo_custom_container
export IMAGE_TAG=v1
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
docker build -f Dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI


export BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
export REGION="us-west1-b"

# gsutil mb gs://${BUCKET_NAME}
#gsutil cp alekseyv-scalableai-dev-077efe757ef6.json gs://alekseyv-scalableai-dev-private-bucket/criteo

CLUSTER_NAME='criteo-cluster'
CURRENT_DATE_UTC=`date --utc -Iseconds`
python scripts/render_template.py $@ | kubectl create -f -

echo "submitted job to kubernetes cluster"
echo ""
echo "log read command:"
echo "gcloud logging read 'resource.type=\"container\"
resource.labels.cluster_name=\"${CLUSTER_NAME}\"
resource.labels.namespace_id=\"default\"
resource.labels.project_id=\"${PROJECT_ID}\"
resource.labels.zone:\"us-central1-a\"
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
resource.labels.zone:\"us-central1-a\"
resource.labels.container_name=\"tensorflow\"
timestamp>=\"${CURRENT_DATE_UTC}\"
" --limit 1000000000000 --order asc --format 'value(resource.labels.pod_id, jsonPayload.message, textPayload)' > logfile.txt
cat logfile.txt | sed '/^$/d'
if [[ $(cat logfile.txt | head -n 5 | wc -l) -ne 0 ]]; then
    CURRENT_DATE_UTC=`date --utc -Iseconds`;
else
    kubectl get events --sort-by=.metadata.creationTimestamp | head -n 3 | tail -n 2
fi

done
