# Distributed training for Criteo dataset on GCP using BigQuery reader

## Overview

## Sample Structure

* [trainer](./trainer) directory: containing the training package to be submitted to AI Platform
  * [__init__py](./trainer/__init__.py) which is an empty file. It is needed to make this directory a Python package.
  * [task.py](./trainer/task.py) contains the training code. It create a simple dummy linear dataset
  and trains a linear regression model with scikit-learn and saves the trained model
  object in a directory (local or GCS) given by the user. 
* [scripts](./scripts) directory: command-line scripts to train the model locally or on AI Platform.
  We recommend to run the scripts in this directory in the following order, and use
  the `source` command to run them, in order to export the environment variables at each step:
  * [train-local.sh](./scripts/train-local.sh) trains the model locally using `gcloud`. It is always a
  good idea to try and train the model locally for debugging, before submitting it to AI Platform.
  * [train-cloud.sh](./scripts/train-cloud.sh) submits a training job to AI Platform.
  * [deploy.sh](./scripts/deploy.sh) creates a model resource, and a model version for the newly trained model.
  * [cleanup.sh](./scripts/cleanup.sh) deletes all the resources created in this tutorial.
* [prediction](./prediction) containing a Python sample code to invoke the model for prediction.
  * [predict.py](./prediction/predict.py) invokes the model for some predictions.
* [setup.py](./setup.py): containing all the required Python packages for this tutorial.


## Running the Sample

TODO: update

## Explaining Key Elements

In this section, we'll highlight the main elements of this sample.

### [task.py](./trainer/trainer.py)

In this sample we are not passing the input dataset as a parameter. However, we need
to save the trained model. To keep things simple, the code expects one argument
to be passed to the code: the path to to store the model in. In other examples, we will
be using `argparse` to process the input arguments. However, in this sample, we simply
read the input argument from `sys.argv[1]`.

Also note that we save the model as `model.joblib` which is
the name that AI Platform expects for models saved with `joblib` to have.

Finally, we are using `tf.gfile` from TensorFlow to upload the model to GCS. It does
not mean we are actually using TensorFlow in this sample to train a model. You may 
also use `google.cloud.storage` library for uploading and downloading to/from GCS.
The advantage of using `tf.gfile` is that it works seamlessly whether the file
path is local or a GCS bucket.

### [train-local.sh](./scripts/train-local.sh)

TODO: update
The command to run the training job locally is this:

```bash
gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        ${MODEL_DIR}
```

* `module-name` is the name of the Python file inside the package which runs the training job
* `package-path` determines where the training Python package is.
* `--` this is just a separator. Anyhing after this will be passed to the training job as input argument.
* `${MODEL_DIR}` will be passed to `task.py` as `sys.argv[1]`

### [train-cloud.sh](./scripts/train-cloud.sh)

TODO: update
To submit a training job to AI Platform, the main command is:

```bash
gcloud ai-platform jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --runtime-version=${RUNTIME_VERSION} \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH}  \
        --python-version=${PYTHON_VERSION} \
        -- \
        ${MODEL_DIR}
```

* `${JOB_NAME}` is a unique name for each job. We create one with a timestamp to make it unique each time.
* `scale-tier` is to choose the tier. For this sample, we use BASIC. However, if you need
to use accelerators for instance, or do a distributed training, you will need a different tier. 
