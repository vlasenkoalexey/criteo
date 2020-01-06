from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.python.platform import gfile

import google.cloud.logging

import argparse

LOCATION = 'us'
PROJECT_ID = "alekseyv-scalableai-dev" # TODO: replace with your project name
GOOGLE_APPLICATION_CREDENTIALS = "alekseyv-scalableai-dev-077efe757ef6.json" # TODO: replace with your key name
GOOGLE_APPLICATION_CREDENTIALS_GCS_BUCKET = 'gs://alekseyv-scalableai-dev-private-bucket/criteo' # TODO: replace with the path to the GCS bucket your project has access to

DATASET_ID = 'criteo_kaggle'

BATCH_SIZE = 128
EPOCHS = 5

FULL_TRAIN_DATASET_SIZE = 36670642 # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train`
SMALL_TRAIN_DATASET_SIZE = 366715  # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train_small`

TRAIN_LOCATION_TYPE_VALUES = 'local cloud'
TRAIN_LOCATION_TYPE = Enum('TRAIN_LOCATION_TYPE', TRAIN_LOCATION_TYPE_VALUES)
TRAIN_LOCATION = TRAIN_LOCATION_TYPE.local

# https://www.tensorflow.org/guide/distributed_training
DISTRIBUTION_STRATEGY_TYPE_VALUES = 'tf.distribute.MirroredStrategy tf.distribute.experimental.ParameterServerStrategy ' \
  'tf.distribute.experimental.MultiWorkerMirroredStrategy tf.distribute.experimental.CentralStorageStrategy'
TRAINING_FUNCTION_VALUES = 'train_keras_sequential train_keras_functional train_keras_functional_wide_and_deep ' \
  'train_keras_to_estimator_functional train_keras_to_estimator_sequential train_estimator train_estimator_wide_and_deep'

DATASET_SIZE_TYPE = Enum('DATASET_SIZE_TYPE', 'full small')
DATASET_SIZE = DATASET_SIZE_TYPE.small

DATASET_SOURCE_TYPE = Enum('DATASET_SOURCE_TYPE', 'bq gcs')
DATASET_SOURCE = DATASET_SOURCE_TYPE.bq

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

def get_vocabulary_size_dict():
  client = bigquery.Client(project=PROJECT_ID)
  query = """
    SELECT
    COUNT(DISTINCT cat1) as cat1,
    COUNT(DISTINCT cat2) as cat2,
    COUNT(DISTINCT cat3) as cat3,
    COUNT(DISTINCT cat4) as cat4,
    COUNT(DISTINCT cat5) as cat5,
    COUNT(DISTINCT cat6) as cat6,
    COUNT(DISTINCT cat7) as cat7,
    COUNT(DISTINCT cat8) as cat8,
    COUNT(DISTINCT cat9) as cat9,
    COUNT(DISTINCT cat10) as cat10,
    COUNT(DISTINCT cat11) as cat11,
    COUNT(DISTINCT cat12) as cat12,
    COUNT(DISTINCT cat13) as cat13,
    COUNT(DISTINCT cat14) as cat14,
    COUNT(DISTINCT cat15) as cat15,
    COUNT(DISTINCT cat16) as cat16,
    COUNT(DISTINCT cat17) as cat17,
    COUNT(DISTINCT cat18) as cat18,
    COUNT(DISTINCT cat19) as cat19,
    COUNT(DISTINCT cat20) as cat20,
    COUNT(DISTINCT cat21) as cat21,
    COUNT(DISTINCT cat22) as cat22,
    COUNT(DISTINCT cat23) as cat23,
    COUNT(DISTINCT cat24) as cat24,
    COUNT(DISTINCT cat25) as cat25,
    COUNT(DISTINCT cat26) as cat26
    FROM
      `alekseyv-scalableai-dev.criteo_kaggle.days`
  """
  query_job = client.query(
      query,
      location=LOCATION,
  )  # API request - starts the query

  df = query_job.to_dataframe()
  dictionary = dict((field[0], df[field[0]][0]) for field in df.items())
  return dictionary

def get_mean_and_std_dicts():
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

def transform_row(row_dict, mean_dict, std_dict):
  dict_without_label = dict(row_dict)
  label = dict_without_label.pop('label')
  for field in CSV_SCHEMA:
    if (field.name.startswith('int')):
        if dict_without_label[field.name] != 0:
            value = float(dict_without_label[field.name])
            dict_without_label[field.name] = (value - mean_dict[field.name]) / std_dict[field.name]
        else:
            dict_without_label[field.name] = 0.0 # don't use normalized 0 value for nulls
  return (dict_without_label, label)

def read_bigquery(table_name):
  if DATASET_SIZE == DATASET_SIZE_TYPE.small:
    table_name += '_small'

  (mean_dict, std_dict) = get_mean_and_std_dicts()
  requested_streams_count = 10
  tensorflow_io_bigquery_client = BigQueryClient()
  read_session = tensorflow_io_bigquery_client.read_session(
      "projects/" + PROJECT_ID,
      PROJECT_ID, table_name, DATASET_ID,
      list(field.name for field in CSV_SCHEMA),
      list(dtypes.int64 if field.field_type == 'INTEGER'
           else dtypes.string for field in CSV_SCHEMA),
      requested_streams=requested_streams_count)

  # manually sharding output instaead of using return read_session.parallel_read_rows()
  streams = read_session.get_streams()
  # streams_count = len(streams) # does not work for Estimator
  streams_count = tf.size(streams)
  streams_count64 = tf.cast(streams_count, dtype=tf.int64)
  streams_ds = dataset_ops.Dataset.from_tensor_slices(streams).shuffle(buffer_size=streams_count64)
  dataset = streams_ds.interleave(
            read_session.read_rows,
            cycle_length=streams_count64,
            num_parallel_calls=streams_count64)

  transformed_ds = dataset.map (lambda row: transform_row(row, mean_dict, std_dict), num_parallel_calls=streams_count) \
    .shuffle(10000) \
    .batch(BATCH_SIZE) \
    .prefetch(100)

  # TODO: enable once tf.data.experimental.AutoShardPolicy.OFF is available
  # Interleave dataset is not shardable, turning off sharding
  # See https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#dataset_sharding_and_batch_size
  # Instead we are shuffling data.
  # options = tf.data.Options()
  #  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  # return transformed_ds.with_options(options)
  return transformed_ds

def transofrom_row_gcs(row_tuple, mean_dict, std_dict):
    row_dict = dict(zip(list(field.name for field in CSV_SCHEMA) + ['row_hash'], list(row_tuple)))
    row_dict.pop('row_hash')
    return transform_row(row_dict, mean_dict, std_dict)


def _get_file_names(file_pattern):
  if isinstance(file_pattern, list):
    if not file_pattern:
      raise ValueError("File pattern is empty.")
    file_names = []
    for entry in file_pattern:
      file_names.extend(gfile.Glob(entry))
  else:
    file_names = list(gfile.Glob(file_pattern))

  if not file_names:
    raise ValueError("No files match %s." % file_pattern)
  return file_names

def read_gcs(table_name):
  if DATASET_SIZE == DATASET_SIZE_TYPE.small:
    table_name += '_small'
  else:
    table_name += '_full'

  gcs_filename_glob = 'gs://alekseyv-scalableai-dev-public-bucket/criteo_kaggle_from_bq/{}*'.format(table_name)
  file_names = _get_file_names(gcs_filename_glob)
  num_parallel_calls = max(10, len(file_names))
  file_names_ds = dataset_ops.Dataset.from_tensor_slices(file_names).shuffle(buffer_size=20)
  record_defaults = list(tf.int32 if field.name == 'label' else tf.constant(0, dtype=tf.int32) if field.name.startswith('int') else tf.constant('', dtype=tf.string) for field in CSV_SCHEMA) + [tf.string]
  dataset = file_names_ds.interleave(
          lambda file_name: tf.data.experimental.CsvDataset(file_name, record_defaults, field_delim='\t', header=False),
          cycle_length=num_parallel_calls,
          num_parallel_calls=num_parallel_calls)

  (mean_dict, std_dict) = get_mean_and_std_dicts()
  transformed_ds = dataset.map (lambda *row_tuple: transofrom_row_gcs(row_tuple, mean_dict, std_dict)) \
    .shuffle(10000) \
    .batch(BATCH_SIZE) \
    .prefetch(100)
  return transformed_ds

def get_dataset(table_name):
  global DATASET_SOURCE
  return read_gcs(table_name) if DATASET_SOURCE == DATASET_SOURCE_TYPE.gcs else read_bigquery(table_name)

def get_max_steps():
  dataset_size = FULL_TRAIN_DATASET_SIZE if DATASET_SIZE == DATASET_SIZE_TYPE.full else SMALL_TRAIN_DATASET_SIZE
  return EPOCHS * dataset_size // BATCH_SIZE

def create_categorical_feature_column(categorical_vocabulary_size_dict, key):
  hash_bucket_size = min(categorical_vocabulary_size_dict[key], 100000)
  # TODO: consider using categorical_column_with_vocabulary_list
  categorical_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
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

def create_linear_feature_columns():
  return list(tf.feature_column.numeric_column(field.name, dtype=tf.dtypes.float32)  for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label')

def create_categorical_feature_columns(categorical_vocabulary_size_dict):
  return list(create_categorical_feature_column(categorical_vocabulary_size_dict, key) for key, _ in categorical_vocabulary_size_dict.items())

def create_feature_columns(categorical_vocabulary_size_dict):
  feature_columns = []
  feature_columns.extend(create_linear_feature_columns())
  feature_columns.extend(create_categorical_feature_columns(categorical_vocabulary_size_dict))
  return feature_columns

def create_input_layer(categorical_vocabulary_size_dict):
    numeric_feature_columns = create_linear_feature_columns()
    numerical_input_layers = {
       feature_column.name: tf.keras.layers.Input(name=feature_column.name, shape=(1,), dtype=tf.float32)
       for feature_column in numeric_feature_columns
    }
    categorical_feature_columns = create_categorical_feature_columns(categorical_vocabulary_size_dict)
    categorical_input_layers = {
       feature_column.categorical_column.name: tf.keras.layers.Input(name=feature_column.categorical_column.name, shape=(), dtype=tf.string)
       for feature_column in categorical_feature_columns
    }
    input_layers = numerical_input_layers.copy()
    input_layers.update(categorical_input_layers)

    return (input_layers, numeric_feature_columns + categorical_feature_columns)

def create_keras_model_functional():
    categorical_vocabulary_size_dict = get_vocabulary_size_dict()
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
    logging.info("model: " + str(model.summary()))
    return model

def create_keras_model_functional_wide_and_deep():
    categorical_vocabulary_size_dict = get_vocabulary_size_dict()
    (feature_layer_inputs, feature_columns) = create_input_layer(categorical_vocabulary_size_dict)
#    linear_feature_columns=create_linear_feature_columns()
    categorical_feature_columns=create_categorical_feature_columns(categorical_vocabulary_size_dict)

    wide = tf.keras.layers.DenseFeatures(categorical_feature_columns)(feature_layer_inputs)

    deep = tf.keras.layers.DenseFeatures(feature_columns)(feature_layer_inputs)
    deep = tf.keras.layers.Dense(2560, activation=tf.nn.relu)(deep)
    deep = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(deep)
    deep = tf.keras.layers.Dense(256, activation=tf.nn.relu)(deep)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(
      tf.keras.layers.concatenate([deep, wide]))

    outputs = tf.squeeze(outputs, -1)
    inputs=[v for v in feature_layer_inputs.values()]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile Keras model
    model.compile(
      # cannot use Adagrad with mirroredstartegy https://github.com/tensorflow/tensorflow/issues/19551
      #optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy'])
    logging.info("model: " + str(model.summary()))
    return model

def create_keras_model_sequential():
  categorical_vocabulary_size_dict = get_vocabulary_size_dict()
  feature_columns = create_feature_columns(categorical_vocabulary_size_dict)
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
  return model

def train_and_evaluate_keras_model(model, model_dir):
  dataset_size = FULL_TRAIN_DATASET_SIZE if DATASET_SIZE == DATASET_SIZE_TYPE.full else SMALL_TRAIN_DATASET_SIZE
  logging.info('training datset size: {}'.format(dataset_size))
  training_ds = get_dataset('train')

  log_dir= os.path.join(model_dir, "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1, profile_batch=0)

  checkpoints_dir= os.path.join(model_dir, "checkpoints")
  if not os.path.exists(checkpoints_dir):
      os.makedirs(checkpoints_dir)
  checkpoints_file_path = checkpoints_dir + "/epochs:{epoch:03d}-accuracy:{accuracy:.3f}.hdf5"
  # crashing https://github.com/tensorflow/tensorflow/issues/27688
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_file_path, verbose=1, mode='max')
  fit_verbosity = 1 if TRAIN_LOCATION == TRAIN_LOCATION_TYPE.local else 2
  model.fit(training_ds, epochs=EPOCHS, verbose=fit_verbosity, callbacks=[tensorboard_callback, checkpoint_callback])
  eval_ds = get_dataset('test')
  loss, accuracy = model.evaluate(eval_ds)
  logging.info("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))

def train_keras_functional_model_to_estimator(strategy, model, model_dir):
    logging.info('training for {} steps'.format(get_max_steps()))
    config = tf.estimator.RunConfig(
            train_distribute=strategy,
            eval_distribute=strategy)
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
    # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
    tf.estimator.train_and_evaluate(
        keras_estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train').repeat(EPOCHS), max_steps=get_max_steps()),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))

def train_keras_sequential(strategy, model_dir):
  train_and_evaluate_keras_model(create_keras_model_sequential(), model_dir)

def train_keras_functional(strategy, model_dir):
  train_and_evaluate_keras_model(create_keras_model_functional(), model_dir)

def train_keras_functional_wide_and_deep(strategy, model_dir):
  train_and_evaluate_keras_model(create_keras_model_functional_wide_and_deep(), model_dir)

def train_keras_to_estimator_sequential(strategy, model_dir):
  train_keras_functional_model_to_estimator(strategy, create_keras_model_sequential(), model_dir)

def train_keras_to_estimator_functional(strategy, model_dir):
  train_keras_functional_model_to_estimator(strategy, create_keras_model_functional(), model_dir)

def train_estimator(strategy, model_dir):
  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig(
          train_distribute=strategy,
          eval_distribute=strategy)
  categorical_vocabulary_size_dict = get_vocabulary_size_dict()
  feature_columns = create_feature_columns(categorical_vocabulary_size_dict)
  estimator = tf.estimator.DNNClassifier(
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      feature_columns=feature_columns,
      hidden_units=[2560, 1024, 256],
      model_dir=model_dir,
      config=config,
      n_classes=2)
  logging.info('training estimator')
  # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
  # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train').repeat(EPOCHS), max_steps=get_max_steps()),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('>>> done evaluating estimtor model')

def train_estimator_wide_and_deep(strategy, model_dir):
  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig(
          train_distribute=strategy,
          eval_distribute=strategy)
  categorical_vocabulary_size_dict = get_vocabulary_size_dict()
  linear_feature_columns=create_linear_feature_columns()
  categorical_feature_columns=create_categorical_feature_columns(categorical_vocabulary_size_dict)
  estimator = tf.estimator.DNNLinearCombinedClassifier(
      dnn_optimizer=tf.optimizers.SGD(learning_rate=0.05),
      linear_optimizer=tf.optimizers.SGD(learning_rate=0.05),
      linear_feature_columns=linear_feature_columns,
      dnn_feature_columns=linear_feature_columns + categorical_feature_columns,
      dnn_hidden_units=[2560, 1024, 256],
      model_dir=model_dir,
      config=config,
      n_classes=2)
  logging.info('training estimator')
  # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
  # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
  result = tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train').repeat(EPOCHS), max_steps=get_max_steps()),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('>>> done evaluating estimtor model: ' + str(result))

def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--train-location',
        help='where to train model - locally or in the cloud',
        choices=TRAIN_LOCATION_TYPE_VALUES.split(' '),
        default='local')

    args_parser.add_argument(
        '--model-name',
        help='model name, not used.')

    args_parser.add_argument(
        '--job-dir',
        help='folder or GCS location to write checkpoints and export models.',
        required=True)

    args_parser.add_argument(
        '--distribution-strategy',
        help='Distribution strategy to use.',
        choices=DISTRIBUTION_STRATEGY_TYPE_VALUES.split(' '))

    args_parser.add_argument(
        '--training-function',
        help='Training function.',
        choices=TRAINING_FUNCTION_VALUES.split(' '),
        default='train_keras_sequential')

    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=128)

    args_parser.add_argument(
        '--dataset-size',
        help='Size of training set',
        choices=['full', 'small'],
        default='small')

    args_parser.add_argument(
        '--dataset-source',
        help='Dataset source.',
        choices=['bq', 'gcs'],
        default='bq')

    args_parser.add_argument(
        '--num-epochs',
        help='Maximum number of training data epochs on which to train.',
        default=2,
        type=int,
    )

    return args_parser.parse_args()

def setup_environment(args):
  global TRAIN_LOCATION
  #logging.warning(os.system('pip list'))
  #logging.warning(os.system('env'))
  #logging.warning('python version: ' + str(sys.version))
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
  os.environ['PROJECT_ID'] = PROJECT_ID

  TF_CONFIG = os.environ.get('TF_CONFIG')
  if (TRAIN_LOCATION == TRAIN_LOCATION_TYPE.local):
    # see https://stackoverflow.com/questions/58868459/tensorflow-assertionerror-on-fit-method
    logging.warning('training locally')
    if TF_CONFIG:
      logging.warning('removing TF_CONFIG')
      os.environ.pop('TF_CONFIG')
  else:
    logging.warning('training in cloud')
    os.system('gsutil cp {}/{} .'.format(GOOGLE_APPLICATION_CREDENTIALS_GCS_BUCKET, GOOGLE_APPLICATION_CREDENTIALS))
    os.environ[ 'GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd() + '/' + GOOGLE_APPLICATION_CREDENTIALS

  if TF_CONFIG and '"master"' in TF_CONFIG and args.distribution_strategy:
    # If distribution strategy is not set, don't replace 'master' -> 'chief',
    # otherwise system will assume that environment works in distributed setting and
    # will expect to be executed in distribution strategy scope.
    # See b/147248890 and
    # https://github.com/tensorflow/tensorflow/blob/64c3d382cadf7bbe8e7e99884bede8284ff67f56/tensorflow/python/distribute/multi_worker_util.py#L235
    # Fixed in TF2.1rc2 https://github.com/tensorflow/tensorflow/commit/0390084145761a1d4da3be2bec8c56a28399db14
    logging.warning('TF_CONFIG before modification:' + str(os.environ['TF_CONFIG']))
    TF_CONFIG = TF_CONFIG.replace('"master"', '"chief"')
    os.environ['TF_CONFIG'] = TF_CONFIG

  if TF_CONFIG:
    logging.warning('TF_CONFIG from env:' + str(os.environ['TF_CONFIG']))

def main():
    global BATCH_SIZE
    global EPOCHS
    global TRAIN_LOCATION
    global DATASET_SOURCE
    global DATASET_SIZE

    args = get_args()
    # https://github.com/tensorflow/tensorflow/issues/34568
    # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model_with_multiworkermirroredstrategy
    # Currently there is a limitation in MultiWorkerMirroredStrategy where TensorFlow ops need to be created after the instance of strategy is created.
    distribution_strategy = None
    if args.distribution_strategy:
        distribution_strategy = eval(args.distribution_strategy)()

    logging_client = google.cloud.logging.Client()
    logging_client.setup_logging()
    logging.getLogger().setLevel(logging.INFO)
    logging.info('trainer started')
    logging.info("tf.test.is_gpu_available(): " + str(tf.test.is_gpu_available()))
    logging.info("device_lib.list_local_devices(): " + str(device_lib.list_local_devices()))

    logging.info('trainer arguments: ' + str(args))

    TRAIN_LOCATION = TRAIN_LOCATION_TYPE[args.train_location]
    logging.info('train_location: ' + str(TRAIN_LOCATION))
    DATASET_SOURCE = DATASET_SOURCE_TYPE[args.dataset_source]
    logging.info('dataset_source: ' + str(DATASET_SOURCE))
    DATASET_SIZE = DATASET_SIZE_TYPE[args.dataset_size]
    logging.info('dataset_size: ' + str(DATASET_SIZE))

    logging.info('distribution_strategy: ' + str(type(distribution_strategy)))

    model_dir = args.job_dir
    if TRAIN_LOCATION == TRAIN_LOCATION_TYPE.cloud and os.environ.get('HOSTNAME'):
      model_dir = os.path.join(model_dir, os.environ.get('HOSTNAME'))
    model_dir = os.path.join(model_dir, args.training_function, 'model.joblib')
    logging.info('Model will be saved to "%s..."', model_dir)

    training_function = getattr(sys.modules[__name__], args.training_function)
    logging.info('training_function: ' + str(training_function))

    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs

    setup_environment(args)

    if not args.distribution_strategy:
      logging.info('no distribution_strategy')
      training_function(None, model_dir)
    else:
      if 'estimator' in args.training_function:
        logging.info('args.training_function:' + args.training_function)
        logging.info('distribution_strategy not in scope: ' + str(type(distribution_strategy)))
        training_function(distribution_strategy, model_dir)
      else:
        with distribution_strategy.scope():
          logging.info('distribution_strategy in scope: ' + str(type(distribution_strategy)))
          training_function(distribution_strategy, model_dir)

if __name__ == '__main__':
    main()
