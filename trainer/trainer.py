from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import sys
import os.path
import logging

import os
from six.moves import urllib
import tempfile

import math
import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum

import datetime
from tensorflow import keras
from tensorflow.keras.callbacks import *

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
from tensorflow.python.client import device_lib

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops

import google.cloud.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("job-dir", "", "Job directory")

LOCATION = 'us'
PROJECT_ID = "alekseyv-scalableai-dev"
GOOGLE_APPLICATION_CREDENTIALS = "alekseyv-scalableai-dev-077efe757ef6.json"

DATASET_ID = 'criteo_kaggle'

BATCH_SIZE = 128

TARGET_TYPE = Enum('TARGET_TYPE', 'local cloud')

CSV_SCHEMA = [
      bigquery.SchemaField("label", "INTEGER", mode='REQUIRED'),
      bigquery.SchemaField("int1", "INTEGER"),
      bigquery.SchemaField("int2", "INTEGER"),
      bigquery.SchemaField("int3", "INTEGER"),
      bigquery.SchemaField("int4", "INTEGER"),
      bigquery.SchemaField("int5", "INTEGER"),
      bigquery.SchemaField("int6", "INTEGER"),
      bigquery.SchemaField("int7", "INTEGER"),
      bigquery.SchemaField("int8", "INTEGER"),
      bigquery.SchemaField("int9", "INTEGER"),
      bigquery.SchemaField("int10", "INTEGER"),
      bigquery.SchemaField("int11", "INTEGER"),
      bigquery.SchemaField("int12", "INTEGER"),
      bigquery.SchemaField("int13", "INTEGER"),
      bigquery.SchemaField("cat1", "STRING"),
      bigquery.SchemaField("cat2", "STRING"),
      bigquery.SchemaField("cat3", "STRING"),
      bigquery.SchemaField("cat4", "STRING"),
      bigquery.SchemaField("cat5", "STRING"),
      bigquery.SchemaField("cat6", "STRING"),
      bigquery.SchemaField("cat7", "STRING"),
      bigquery.SchemaField("cat8", "STRING"),
      bigquery.SchemaField("cat9", "STRING"),
      bigquery.SchemaField("cat10", "STRING"),
      bigquery.SchemaField("cat11", "STRING"),
      bigquery.SchemaField("cat12", "STRING"),
      bigquery.SchemaField("cat13", "STRING"),
      bigquery.SchemaField("cat14", "STRING"),
      bigquery.SchemaField("cat15", "STRING"),
      bigquery.SchemaField("cat16", "STRING"),
      bigquery.SchemaField("cat17", "STRING"),
      bigquery.SchemaField("cat18", "STRING"),
      bigquery.SchemaField("cat19", "STRING"),
      bigquery.SchemaField("cat20", "STRING"),
      bigquery.SchemaField("cat21", "STRING"),
      bigquery.SchemaField("cat22", "STRING"),
      bigquery.SchemaField("cat23", "STRING"),
      bigquery.SchemaField("cat24", "STRING"),
      bigquery.SchemaField("cat25", "STRING"),
      bigquery.SchemaField("cat26", "STRING")
  ]

# hack because model_to_estimator does not understand input feature names, see
# https://cs.corp.google.com/piper///depot/google3/third_party/tensorflow_estimator/python/estimator/keras.py?rcl=282034610&l=151
KERAS_TO_ESTIMATOR_FEATURE_NAMES = {}
for i in range(0, len(CSV_SCHEMA)):
  if i != 0:  # skip label
    KERAS_TO_ESTIMATOR_FEATURE_NAMES[CSV_SCHEMA[i].name] = 'input_{}'.format(i)

print('KERAS_TO_ESTIMATOR_FEATURE_NAMES')
print(KERAS_TO_ESTIMATOR_FEATURE_NAMES)

def get_mean_and_std_dicts():
  #client = bigquery.Client(location="US", project=PROJECT_ID)
  client = bigquery.Client(project=PROJECT_ID)
  query = """
    select
    AVG(int1) as avg_int1, STDDEV(int1) as std_int1,
    AVG(int2) as avg_int2, STDDEV(int2) as std_int2,
    AVG(int3) as avg_int3, STDDEV(int3) as std_int3,
    AVG(int4) as avg_int4, STDDEV(int4) as std_int4,
    AVG(int5) as avg_int5, STDDEV(int5) as std_int5,
    AVG(int6) as avg_int6, STDDEV(int6) as std_int6,
    AVG(int7) as avg_int7, STDDEV(int7) as std_int7,
    AVG(int8) as avg_int8, STDDEV(int8) as std_int8,
    AVG(int9) as avg_int9, STDDEV(int9) as std_int9,
    AVG(int10) as avg_int10, STDDEV(int10) as std_int10,
    AVG(int11) as avg_int11, STDDEV(int11) as std_int11,
    AVG(int12) as avg_int12, STDDEV(int12) as std_int12,
    AVG(int13) as avg_int13, STDDEV(int13) as std_int13
    from `alekseyv-scalableai-dev.criteo_kaggle.days`
  """
  query_job = client.query(
      query,
      location="US",
  )  # API request - starts the query

  df = query_job.to_dataframe()

  mean_dict = dict((field[0].replace('avg_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('avg'))
  std_dict = dict((field[0].replace('std_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('std'))
  return (mean_dict, std_dict)

def transofrom_row(row_dict, mean_dict, std_dict):
  dict_without_label = row_dict.copy()
  label = dict_without_label.pop('label')
  for field in CSV_SCHEMA:
    if (field.name.startswith('int')):
        if dict_without_label[field.name] == 0:
            value = float(dict_without_label[field.name])
            dict_without_label[field.name] = (value - mean_dict[field.name]) / std_dict[field.name]
        else:
            dict_without_label[field.name] = 0.0 # don't use normalized 0 value for nulls

  #dict_with_esitmator_keys = { KERAS_TO_ESTIMATOR_FEATURE_NAMES[k]:v for k,v in dict_without_label.items() }
  dict_with_esitmator_keys = { k:v for k,v in dict_without_label.items() }

  return (dict_with_esitmator_keys, label)

def read_bigquery(dataset_id, table_name):
  (mean_dict, std_dict) = get_mean_and_std_dicts()
  tensorflow_io_bigquery_client = BigQueryClient()
  read_session = tensorflow_io_bigquery_client.read_session(
      "projects/" + PROJECT_ID,
      PROJECT_ID, table_name, dataset_id,
      list(field.name for field in CSV_SCHEMA),
      list(dtypes.int64 if field.field_type == 'INTEGER'
           else dtypes.string for field in CSV_SCHEMA),
      requested_streams=10)

  #dataset = read_session.parallel_read_rows()

  streams = read_session.get_streams()
  tf.print('bq streams: !!!!!!!!!!!!!!!!!!!!!!')
  tf.print(streams)
  streams_count = 10 # len(streams)
  #streams_count = read_session.get_streams().shape
  tf.print('big query read session returned {} streams'.format(streams_count))

  streams_ds = dataset_ops.Dataset.from_tensor_slices(streams).shuffle(buffer_size=streams_count)
  dataset = streams_ds.interleave(
            read_session.read_rows,
            cycle_length=streams_count,
            num_parallel_calls=streams_count)
  transformed_ds = dataset.map (lambda row: transofrom_row(row, mean_dict, std_dict), num_parallel_calls=streams_count).prefetch(10000)

  # Interleave dataset is not shardable, turning off sharding
  # See https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#dataset_sharding_and_batch_size
  # Instead we are shuffling data.
  options = tf.data.Options()
#  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  result = transformed_ds.with_options(options)
  tf.print(str(result))
  return result



def create_categorical_feature_column(categorical_vocabulary_size_dict, key):
  hash_bucket_size = min(categorical_vocabulary_size_dict[key], 100000)
  # TODO: consider using categorical_column_with_vocabulary_list
  categorical_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
    #KERAS_TO_ESTIMATOR_FEATURE_NAMES[key],
    key,
    hash_bucket_size,
    dtype=tf.dtypes.string
  )
  if hash_bucket_size < 10:
    return tf.feature_column.indicator_column(categorical_feature_column)

  embedding_feature_column = tf.feature_column.embedding_column(
      categorical_feature_column,
      int(min(50, math.floor(6 * hash_bucket_size**0.25))))
  return embedding_feature_column

def create_feature_columns(categorical_vocabulary_size_dict):
  feature_columns = []
  #feature_columns.extend(list(tf.feature_column.numeric_column(KERAS_TO_ESTIMATOR_FEATURE_NAMES[field.name], dtype=tf.dtypes.float32)  for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label'))
  feature_columns.extend(list(create_categorical_feature_column(categorical_vocabulary_size_dict, key) for key, _ in categorical_vocabulary_size_dict.items()))
  return feature_columns

def create_keras_model_sequential():
  categorical_vocabulary_size_dict = get_vocabulary_size_dict()
  feature_columns = create_feature_columns(categorical_vocabulary_size_dict)
  print("categorical_vocabulary_size_dict: " + str(categorical_vocabulary_size_dict))
  feature_layer = tf.keras.layers.DenseFeatures(feature_columns, name="feature_layer")
  Dense = tf.keras.layers.Dense
  model = tf.keras.Sequential(
  [
      feature_layer,
      Dense(2560, activation=tf.nn.relu),
      Dense(1024, activation=tf.nn.relu),
      Dense(256, activation=tf.nn.relu),
      Dense(1, activation=tf.nn.sigmoid)
  ])

  # Compile Keras model
  model.compile(
      # cannot use Adagrad with mirroredstartegy https://github.com/tensorflow/tensorflow/issues/19551
      #optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy'])
  # HACK: https://b.corp.google.com/issues/114035274
  #model._is_graph_network = True
  #logging.info(model.summary())
  return model

def train_keras_model_sequential(model_dir):
  logging.info('training keras model')
  #strategy = tf.distribute.experimental.ParameterServerStrategy()
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # doesn't work because of https://b.corp.google.com/issues/142700914
  #strategy = tf.distribute.MirroredStrategy()
  #strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
  with strategy.scope():
    model = create_keras_model_sequential()
    #training_ds = read_bigquery('criteo_kaggle','days_strings').take(1000000).shuffle(10000).batch(BATCH_SIZE).prefetch(100)
    #training_ds = read_bigquery('criteo_kaggle','days').skip(100000).take(50000).shuffle(10000).batch(BATCH_SIZE)
    training_ds = read_bigquery('criteo_kaggle','days').take(1000000).shuffle(10000).batch(BATCH_SIZE)
    print('checking dataset')

    log_dir= model_dir + "/" + os.environ['HOSTNAME'] + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1, profile_batch=0)

    checkpoints_dir = model_dir + "/" + os.environ['HOSTNAME'] + "/checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    checkpoints_file_path = checkpoints_dir + "/epochs:{epoch:03d}-accuracy:{accuracy:.3f}.hdf5"
    # crashing https://github.com/tensorflow/tensorflow/issues/27688
    #checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_file_path, verbose=1, mode='max')

    fit_verbosity = 1 if TARGET == TARGET_TYPE.local else 2
    model.fit(training_ds, epochs=2, verbose=fit_verbosity,
    callbacks=[tensorboard_callback]
    )

  return model

def evaluate_keras_model(model):
  logging.info('evaluating keras model')
  eval_ds = read_bigquery('criteo_kaggle','days').skip(100000).take(50 * BATCH_SIZE).batch(BATCH_SIZE)
  loss, accuracy = model.evaluate(eval_ds)
  logging.info("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))

def input_fn():
  training_ds = read_bigquery('criteo_kaggle','days').take(1000000).shuffle(10000).batch(BATCH_SIZE)
  return training_ds

def create_input_layer(categorical_vocabulary_size_dict):
    numeric_feature_columns = list(tf.feature_column.numeric_column(field.name, dtype=tf.dtypes.float32)  for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label')
    numerical_input_layers = {
       feature_column.name: tf.keras.layers.Input(name=feature_column.name, shape=(1,), dtype=tf.float32)
       for feature_column in numeric_feature_columns
    }
    categorical_feature_columns = list(create_categorical_feature_column(categorical_vocabulary_size_dict, key) for key, _ in categorical_vocabulary_size_dict.items())
    #print("categorical_feature_columns: " + str(categorical_feature_columns))
    categorical_input_layers = {
       feature_column.categorical_column.name: tf.keras.layers.Input(name=feature_column.categorical_column.name, shape=(), dtype=tf.string)
       for feature_column in categorical_feature_columns
    }
    #print("categorical_input_layers: " + str(categorical_input_layers))
    input_layers = numerical_input_layers.copy()
    input_layers.update(categorical_input_layers)

    return (input_layers, numeric_feature_columns + categorical_feature_columns)

def create_keras_model_functional(categorical_vocabulary_size_dict):
    (feature_layer_inputs, feature_columns) = create_input_layer(categorical_vocabulary_size_dict)
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    feature_layer_outputs = feature_layer(feature_layer_inputs)
    x = tf.keras.layers.Dense(2560, activation=tf.nn.relu)(feature_layer_outputs)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
    inputs=[v for v in feature_layer_inputs.values()]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile Keras model
    model.compile(
      # cannot use Adagrad with mirroredstartegy https://github.com/tensorflow/tensorflow/issues/19551
      #optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy'])
    print("model: " + str(model.summary()))
    return model

def train_keras_functional_model_to_estimator(model_dir):
    categorical_vocabulary_size_dict = get_vocabulary_size_dict()
    model = create_keras_model_functional(categorical_vocabulary_size_dict)
    # tf.keras.backend.set_learning_phase(True)
    config = tf.estimator.RunConfig(
            train_distribute=tf.distribute.MirroredStrategy(),
            eval_distribute=tf.distribute.MirroredStrategy())
    # keras_estimator = tf.keras.estimator.model_to_estimator(
    #     keras_model=model, config=config, model_dir=model_dir)
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir)

    logging.info('!!!!!!!!!!!!! training MirroredStrategy on keras_estimator !!!!!!!!!!!!!!!!!!!')
    tf.estimator.train_and_evaluate(
        keras_estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=5),
        eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


def main(argv):
    if len(argv) < 1:
      raise app.UsageError("Too few command-line arguments.")

    #tf.compat.v1.enable_eager_execution()

    model_dir = os.path.join(sys.argv[1], 'model.joblib')
    logging.info('Model will be saved to "%s..."', model_dir)

    # #tf.debugging.set_log_device_placement(True)
    # print("tf.config.experimental.list_logical_devices(GPU): " + str(tf.config.experimental.list_logical_devices('GPU')))
    # print("tf.config.experimental.list_physical_devices(GPU): " + str(tf.config.experimental.list_physical_devices('GPU')))
    # print("device_lib.list_local_devices(): " + str(device_lib.list_local_devices()))
    # print("tf.test.is_gpu_available(): " + str(tf.test.is_gpu_available()))

    dataset = read_bigquery('criteo_kaggle','days').take(10)
    row_index = 0
    for row in dataset.prefetch(10):
      print("row %d: %s" % (row_index, row))
      row_index += 1

    #model = train_keras_model(model_dir)
    #evaluate_keras_model(model)
    #train_keras_functional_model_to_estimator(model_dir)

    logging.warn('>>>>>>>>>>>>>>>>> training using DNNClassifier <<<<<<<<<<<<<<<<<<<')
    categorical_vocabulary_size_dict = get_vocabulary_size_dict()
    feature_columns = create_feature_columns(categorical_vocabulary_size_dict)

    config = tf.estimator.RunConfig(
        train_distribute=tf.distribute.MirroredStrategy(),
        eval_distribute=tf.distribute.MirroredStrategy())

    estimator = tf.estimator.DNNClassifier(
        optimizer=tf.optimizers.SGD(learning_rate=0.05),
        feature_columns=feature_columns,
        hidden_units=[2560, 1024, 256],
        model_dir=model_dir,
        n_classes=2,
        config=config)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=input_fn, max_steps=2),
        eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


if __name__ == '__main__':
  logging_client = google.cloud.logging.Client()
  logging_client.setup_logging()
  logging.getLogger().setLevel(logging.INFO)
  logging.warning('>>>>>>>>>>>>>>>>>>>>>>>>>> app started logging <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
  logging.warning(os.system('env'))
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
  os.environ['PROJECT_ID'] = PROJECT_ID
  logging.warning('python version: ' + str(sys.version))
  logging.warning(os.system('pip list'))
  #print(os.system('pwd'))
  #print(os.system('ls -al'))
  if (os.environ.get('CLOUDSDK_METRICS_COMMAND_NAME') == 'gcloud.ai-platform.local.train'):
    TARGET = TARGET_TYPE.local
    logging.warning('training locally')
    logging.warning('removing TF_CONFIG')
    os.environ.pop('TF_CONFIG')
  else:
    TARGET = TARGET_TYPE.cloud
    logging.warning('training in cloud')
    os.system('gsutil cp gs://alekseyv-scalableai-dev-private-bucket/criteo/alekseyv-scalableai-dev-077efe757ef6.json .')
    os.environ[ "GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + '/' + GOOGLE_APPLICATION_CREDENTIALS
    #os.system('gsutil cp gs://alekseyv-scalableai-dev-private-bucket/criteo/tensorflow_io-0.10.0-cp27-cp27mu-manylinux2010_x86_64.wh .')
    #os.system('pip install --no-deps tensorflow_io-0.10.0-cp27-cp27mu-manylinux2010_x86_64.whl')

  TF_CONFIG = os.environ.get('TF_CONFIG')
  # if TF_CONFIG and '"master"' in TF_CONFIG:
  #   logging.warning('TF_CONFIG before modification:' + str(os.environ['TF_CONFIG']))
  #   os.environ['TF_CONFIG'] = TF_CONFIG.replace('"master"', '"chief"')

  if TF_CONFIG:
    logging.warning('TF_CONFIG:' + str(os.environ['TF_CONFIG']))
  logging.warning(os.system('cat ${GOOGLE_APPLICATION_CREDENTIALS}'))
  app.run(main)

