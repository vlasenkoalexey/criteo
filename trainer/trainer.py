from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging

import os
import tempfile

import time
import math
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
import tensorflow_io as tf_io
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
from tensorflow.python.client import device_lib

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import gfile

import google.cloud.logging

import argparse

class BatchAccuracyAndLossSummaryCallback(tf.keras.callbacks.Callback):
  # TODO: make it dist. strat. compartible
  def __init__(self, log_dir, dataset_size):
    # Callback should only write summaries on the chief when in a Multi-Worker setting.
    self._chief_worker_only = True
    self.update_freq = 50 if dataset_size == 'small' else 1000
  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch
  def on_train_batch_end(self, batch, logs=None):
    if batch % self.update_freq == 0:
      if 'accuracy' in logs:
        tf.summary.scalar('accuracy', logs['accuracy'], batch, description='epoch: {}'.format(self.epoch))
        tf.summary.scalar('accuracy epoch: {}'.format(self.epoch), logs['accuracy'], batch, description='epoch: {}'.format(self.epoch))
      if 'loss' in logs:
        tf.summary.scalar('loss', logs['loss'], batch, description='epoch: {}'.format(self.epoch))
        tf.summary.scalar('loss epoch: {}'.format(self.epoch), logs['loss'], batch, description='epoch: {}'.format(self.epoch))

class TrainTimeCallback(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start_time = datetime.datetime.now()

  def on_epoch_end(self, epoch, logs=None):
    logging.info('\nepoch train time: (hh:mm:ss.ms) {}'.format(datetime.datetime.now() - self.epoch_start_time))
    if not self.params is None:
      if 'steps' in self.params and self.params['steps']:
        epoch_milliseconds = (datetime.datetime.now() - self.epoch_start_time).total_seconds() * 1000
        logging.info('{} ms/step'.format(epoch_milliseconds / self.params['steps']))
        if BATCH_SIZE is not None:
          logging.info('{} microseconds/example'.format(1000 * epoch_milliseconds  / self.params['steps'] / BATCH_SIZE))

  def on_train_begin(self, logs=None):
    self.start_training_time = datetime.datetime.now()

  def on_train_end(self, logs=None):
    logging.info('total train time: (hh:mm:ss.ms) {}'.format(datetime.datetime.now() - self.start_training_time))

LOCATION = 'us'
PROJECT_ID = "alekseyv-scalableai-dev" # TODO: replace with your project name
GOOGLE_APPLICATION_CREDENTIALS = "alekseyv-scalableai-dev-077efe757ef6.json" # TODO: replace with your key name
GOOGLE_APPLICATION_CREDENTIALS_GCS_BUCKET = 'gs://alekseyv-scalableai-dev-private-bucket/criteo' # TODO: replace with the path to the GCS bucket your project has access to

DATASET_ID = 'criteo_kaggle'

BATCH_SIZE = 128
EPOCHS = 5
EMBEDDINGS_MODE_TYPE_VALUES = 'none manual hashbucket vocabular'
EMBEDDINGS_MODE_TYPE = Enum('EMBEDDINGS_MODE_TYPE', 'none manual hashbucket vocabular')
EMBEDDINGS_MODE = EMBEDDINGS_MODE_TYPE.hashbucket

FULL_TRAIN_DATASET_SIZE = 36670642 # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train`
SMALL_TRAIN_DATASET_SIZE = 366715  # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train_small`
FULL_TEST_DATASET_SIZE = 4589420 # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train`
SMALL_TEST_DATASET_SIZE = 50048  # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train_small`

TRAIN_LOCATION_TYPE_VALUES = 'local cloud'
TRAIN_LOCATION_TYPE = Enum('TRAIN_LOCATION_TYPE', TRAIN_LOCATION_TYPE_VALUES)
TRAIN_LOCATION = TRAIN_LOCATION_TYPE.local

# https://www.tensorflow.org/guide/distributed_training
DISTRIBUTION_STRATEGY_TYPE = None
DISTRIBUTION_STRATEGY_TYPE_VALUES = 'tf.distribute.MirroredStrategy tf.distribute.experimental.ParameterServerStrategy ' \
  'tf.distribute.experimental.MultiWorkerMirroredStrategy tf.distribute.experimental.CentralStorageStrategy ' \
  'tf.distribute.experimental.TPUStrategy tf.distribute.OneDeviceStrategy'
TRAINING_FUNCTION_VALUES = 'train_keras_sequential train_keras_functional train_keras_functional_wide_and_deep ' \
  'train_keras_to_estimator_functional train_keras_to_estimator_sequential train_estimator train_estimator_wide_and_deep ' \
  'train_keras_functional_no_feature_layer train_custom_loop_keras_sequential train_custom_loop_keras_model_functional_no_feature_layer'

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

# See https://docs.google.com/spreadsheets/d/1s_QCP4dyO9L9QXVQu0TKrIi767kMwlr8SsfAX5UCNrw/edit#gid=118220263
KEY_CROSSES = [
  ['cat13','cat18'],
  ['cat11', 'cat18'],
  ['cat11', 'cat15'],
  ['cat2', 'cat8'],
  ['cat7', 'cat15'],
  ['cat13', 'cat15'],
  ['cat1', 'cat2'],
  ['cat13', 'cat19'],
  ['cat2', 'cat13'],
  ['cat7', 'cat18'],
  ['cat1', 'cat13'],
  ['cat2', 'cat5'],
  ['cat2', 'cat11'],
  ['cat11', 'cat19'],
  ['cat1', 'cat11'],
  ['cat8', 'cat13']
]

def get_mean_and_std_dicts():
  table_name = 'days' if DATASET_SIZE == DATASET_SIZE_TYPE.full else 'small'
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
    from `alekseyv-scalableai-dev.criteo_kaggle.{table_name}`
  """.format(table_name = table_name)
  query_job = client.query(
      query,
      location=LOCATION,
  )  # API request - starts the query

  df = query_job.to_dataframe()

  mean_dict = dict((field[0].replace('avg_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('avg'))
  std_dict = dict((field[0].replace('std_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('std'))
  return (mean_dict, std_dict)

def get_vocabulary_size_dict():
  table_name = 'days' if DATASET_SIZE == DATASET_SIZE_TYPE.full else 'small'
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
      `alekseyv-scalableai-dev.criteo_kaggle.{table_name}`
  """.format(table_name = table_name)
  query_job = client.query(
      query,
      location=LOCATION,
  )  # API request - starts the query

  df = query_job.to_dataframe()
  dictionary = dict((field[0], df[field[0]][0]) for field in df.items())
  return dictionary

def get_corpus_dict():
  table_name = 'days' if DATASET_SIZE == DATASET_SIZE_TYPE.full else 'small'
  client = bigquery.Client(project=PROJECT_ID)
  query = """
    select
    cat_name,
    cat_value,
    cat_index
    from `alekseyv-scalableai-dev.criteo_kaggle.{table_name}_corpus`
  """.format(table_name = table_name)
  query_job = client.query(
      query,
      location="US",
  )  # API request - starts the query

  df = query_job.to_dataframe()
  corpus = dict()
  for _, row in df.iterrows():
    cat_name = row[0]
    cat_value = row[1]
    cat_index = row[2]
    if not cat_name in corpus:
      corpus[cat_name] = dict()
    if cat_value is None:
      cat_value = ''
    corpus[cat_name][cat_value] = cat_index
  return corpus

def corpus_to_lookuptable(corpus):
  lookup_dict = dict()
  for key, value in corpus.items():
    initializer = tf.lookup.KeyValueTensorInitializer(
      list(value.keys()),
      list(value.values()),
      key_dtype=tf.string,
      value_dtype=tf.int64)
    # cat_index in corpus starts with 1, reserving 0 for out of vocabulary values
    lookup_table = tf.lookup.StaticHashTable(initializer, 0)
    lookup_dict[key] = lookup_table
  return lookup_dict

def get_corpus():
  if EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.manual or EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.vocabular:
    return corpus_to_lookuptable(get_corpus_dict())
  else:
    return dict()

# @tf.function is not really necessary here since function is used in tf.data pipeline
# Use print(tf.autograph.to_code(transform_row.python_function)) to see code generated by AutoGraph
@tf.function
def transform_row(row_dict, mean_dict, std_dict, corpus):
  dict_without_label = dict(row_dict)
  label = dict_without_label.pop('label')
  for field in CSV_SCHEMA:
    if (field.name.startswith('int')):
        # use normalized mean value if data is missing (value will be 0)
        value = float(dict_without_label[field.name])
        dict_without_label[field.name] = (value - mean_dict[field.name]) / std_dict[field.name]
    elif field.name.startswith('cat'):
      if EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.none:
        dict_without_label.pop(field.name)
      elif EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.manual:
        cat = dict_without_label[field.name]
        if cat is None:
          cat = ''
        cat_index = corpus[field.name].lookup(cat)
        if cat_index is None:
          tf.print('not found for {}'.format(field.name))
          cat_index = tf.constant(-1)
        dict_without_label[field.name] = cat_index
  return (dict_without_label, label)

def read_bigquery(table_name):
  if DATASET_SIZE == DATASET_SIZE_TYPE.small:
    table_name += '_small'

  (mean_dict, std_dict) = get_mean_and_std_dicts()
  corpus = get_corpus()
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

  transform_row_function = lambda row: transform_row(row, mean_dict, std_dict, corpus)

  transformed_ds = dataset\
    .batch(BATCH_SIZE) \
    .shuffle(50) \
    .map (transform_row_function, num_parallel_calls=streams_count) \
    .prefetch(50)

  # TODO: enable once tf.data.experimental.AutoShardPolicy.OFF is available
  # Interleave dataset is not shardable, turning off sharding
  # See https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#dataset_sharding_and_batch_size
  # Instead we are shuffling data.
  #return transformed_ds
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  return transformed_ds.with_options(options)

def transofrom_row_gcs(row_tuple, mean_dict, std_dict, corpus):
    row_dict = dict(zip(list(field.name for field in CSV_SCHEMA) + ['row_hash'], list(row_tuple)))
    row_dict.pop('row_hash')
    return transform_row(row_dict, mean_dict, std_dict, corpus)


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

  corpus = get_corpus()
  (mean_dict, std_dict) = get_mean_and_std_dicts()
  transofrom_row_gcs_function = lambda *row_tuple: transofrom_row_gcs(row_tuple, mean_dict, std_dict, corpus)

  transformed_ds = dataset\
    .batch(BATCH_SIZE) \
    .shuffle(50) \
    .map (transofrom_row_gcs_function, num_parallel_calls=num_parallel_calls) \
    .prefetch(50)
  return transformed_ds

def get_dataset(table_name):
  global DATASET_SOURCE
  return read_gcs(table_name) if DATASET_SOURCE == DATASET_SOURCE_TYPE.gcs else read_bigquery(table_name)

def get_training_steps_per_epoch():
  global DATASET_SOURCE
  dataset_size = FULL_TRAIN_DATASET_SIZE if DATASET_SIZE == DATASET_SIZE_TYPE.full else SMALL_TRAIN_DATASET_SIZE
  return dataset_size // BATCH_SIZE

def get_validation_steps_per_epoch():
  global DATASET_SOURCE
  dataset_size = FULL_TEST_DATASET_SIZE if DATASET_SIZE == DATASET_SIZE_TYPE.full else SMALL_TEST_DATASET_SIZE
  return dataset_size // BATCH_SIZE

def get_max_steps():
  global EPOCHS
  return EPOCHS * get_steps_per_epoch()

def create_categorical_feature_column_with_hash_bucket(corpus_dict, key):
  corpus_size = len(corpus_dict[key])
  hash_bucket_size = min(corpus_size, 100000)
  categorical_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=tf.dtypes.string
  )
  logging.info('categorical column %s hash_bucket_size %d', key, hash_bucket_size)
  return categorical_feature_column

def create_categorical_feature_column_with_vocabulary_list(corpus_dict, key):
  corpus_size = len(corpus_dict[key])
  categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key,
    list(corpus_dict[key].keys()),
    dtype=tf.dtypes.string,
    num_oov_buckets=corpus_size
  )
  logging.info('categorical column with vocabular %s corpus_size %d', key, corpus_size)

  return categorical_feature_column

def create_embedding(corpus_dict, key, categorical_feature_column):
  corpus_size = len(corpus_dict[key])
  if corpus_size < 10:
    logging.info('categorical column %s corpus_size %d - creating indicator column', key, corpus_size)
    return tf.feature_column.indicator_column(categorical_feature_column)

  embedding_dimension = int(min(50, math.floor(6 * corpus_size**0.25)))
  embedding_feature_column = tf.feature_column.embedding_column(
      categorical_feature_column,
      embedding_dimension)
  return embedding_feature_column

def create_linear_feature_columns():
  return list(tf.feature_column.numeric_column(field.name, dtype=tf.dtypes.float32)  for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label')

def create_categorical_embeddings_feature_columns(corpus_dict):
  if EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.none:
    return []
  elif EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.hashbucket:
    return list(create_embedding(
      corpus_dict,
      key,
      create_categorical_feature_column_with_hash_bucket(corpus_dict, key))
        for key, _ in corpus_dict.items())
  elif EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.vocabular:
    return list(create_embedding(
      corpus_dict,
      key,
      create_categorical_feature_column_with_vocabulary_list(corpus_dict, key))
        for key, _ in corpus_dict.items())
  else:
    raise ValueError('invalid EMBEDDINGS_MODE: {}'.format(EMBEDDINGS_MODE))

def create_feature_columns():
  corpus_dict = get_corpus_dict()
  feature_columns = []
  feature_columns.extend(create_linear_feature_columns())
  feature_columns.extend(create_categorical_embeddings_feature_columns(corpus_dict))
  return feature_columns

def create_input_layer():
    corpus_dict = get_corpus_dict()
    numeric_feature_columns = create_linear_feature_columns()
    numerical_input_layers = {
       feature_column.name: tf.keras.layers.Input(name=feature_column.name, shape=(1,), dtype=tf.float32)
       for feature_column in numeric_feature_columns
    }
    categorical_feature_columns = create_categorical_embeddings_feature_columns(corpus_dict)
    categorical_input_layers = {
       feature_column.categorical_column.name: tf.keras.layers.Input(name=feature_column.categorical_column.name, shape=(), dtype=tf.string)
       for feature_column in categorical_feature_columns
    }
    input_layers = numerical_input_layers.copy()
    input_layers.update(categorical_input_layers)

    return (input_layers, numeric_feature_columns + categorical_feature_columns)

def create_embedding_from_input(corpus_dict, name, input_layer):
  size = len(corpus_dict[name]) + 2
  dimension =  int(min(50, math.floor(6 * size**0.25)))
  logging.info('embedding name:{} size:{} dim:{}'.format(name, size, dimension))
  embedding = tf.keras.layers.Embedding(size, dimension, name = name + '_embedding', input_length=1)(input_layer)
  return embedding

def create_keras_model_functional():
    (feature_layer_inputs, feature_columns) = create_input_layer()
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
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      #optimizer=tf.optimizers.Adam(),
      #optimizer=tf.optimizers.Adagrad(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy'])
    logging.info("model: " + str(model.summary()))
    return model

def create_keras_model_functional_no_feature_layer():
  corpus_dict = get_corpus_dict()
  categorical_input_with_names = list((field.name, tf.keras.layers.Input(shape=[1], name = field.name, dtype=tf.int32))
    for field in CSV_SCHEMA if field.field_type == 'STRING' and field.name != 'label')
  categorical_inputs = list(input_layer
    for (name, input_layer) in categorical_input_with_names)
  categorical_embeddings = list(create_embedding_from_input(corpus_dict, name, input_layer)
    for (name, input_layer) in categorical_input_with_names)

  numerical_inputs = list(tf.keras.layers.Input(shape=[1], name = field.name, dtype=tf.float32)
    for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label')

  categorical_combined = tf.keras.layers.concatenate(categorical_embeddings)
  x = tf.keras.layers.Flatten()(categorical_combined)
  x = tf.keras.layers.concatenate([x] + numerical_inputs)
  x = tf.keras.layers.Dense(2560, activation=tf.nn.relu)(x)
  x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
  x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)

  outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
  model = tf.keras.Model(inputs=categorical_inputs + numerical_inputs, outputs=outputs)

  # Compile Keras model
  model.compile(
    # cannot use Adagrad with mirroredstartegy https://github.com/tensorflow/tensorflow/issues/19551
    optimizer=tf.optimizers.SGD(learning_rate=0.05),
    #optimizer=tf.optimizers.Adam(),
    #optimizer=tf.optimizers.Adagrad(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])
  logging.info("model: " + str(model.summary()))

  return model

def create_keras_model_functional_wide_and_deep_dontuse():
    (feature_layer_inputs, feature_columns) = create_input_layer()
    categorical_feature_columns=create_categorical_embeddings_feature_columns(get_corpus_dict())

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

def create_keras_model_functional_wide_and_deep():
    if EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.none:
      raise ValueError('embeddings are required for wide and deep model')

    linear_feature_columns = create_linear_feature_columns()
    corpus_dict = get_corpus_dict()
    categorical_feature_columns_and_embeddings = create_categorical_embeddings_feature_columns(corpus_dict)
    categorical_feature_columns = list(tf.feature_column.indicator_column(embedding.categorical_column) for embedding in categorical_feature_columns_and_embeddings)
    embeddings = list(embedding for embedding in categorical_feature_columns_and_embeddings)

    crossed_columns = list(tf.feature_column.indicator_column(tf.feature_column.crossed_column(
      keys,
      min(max(len(corpus_dict[keys[0]]), len(corpus_dict[keys[1]])), 100000))) for keys in KEY_CROSSES)

    deep_model_feature_columns = embeddings + linear_feature_columns
    dense_feature_layer = tf.keras.layers.DenseFeatures(deep_model_feature_columns)
    Dense = tf.keras.layers.Dense
    dense_model = tf.keras.Sequential(
    [
        dense_feature_layer,
        Dense(2560, activation=tf.nn.relu),
        Dense(1024, activation=tf.nn.relu),
        Dense(256, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.sigmoid)
    ])

    wide_model_feature_columns = categorical_feature_columns + crossed_columns
    # no sparse alternative, see https://github.com/tensorflow/community/pull/188/
    # https://cs/piper///depot/google3/learning/tfx/users/tfx/examples/tfx_keras_widedeep/models/premade_widedeep/model.py;rcl=291050847;l=100
    sparse_feature_layer = tf.keras.layers.DenseFeatures(wide_model_feature_columns)

    linear_model = tf.keras.Sequential(
    [
      sparse_feature_layer,
      Dense(1, activation=tf.nn.sigmoid)
    ])

    wide_deep_model = tf.keras.experimental.WideDeepModel(linear_model, dense_model)
    #wide_deep_model(deep_model_feature_columns) # todo change
    linear_opt = tf.keras.optimizers.RMSprop()
    dnn_opt = tf.keras.optimizers.Adam()

    wide_deep_model.compile(
        optimizer=[linear_opt, dnn_opt],
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    return wide_deep_model

def create_keras_model_sequential():
  feature_columns = create_feature_columns()

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

  logging.info('compiling sequential keras model')
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

  #log_dir= os.path.join(model_dir, "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  log_dir= os.path.join(model_dir, "logs")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=1,
    profile_batch=min(EPOCHS, 2))

  checkpoints_dir= os.path.join(model_dir, "checkpoints")
  # crashing https://github.com/tensorflow/tensorflow/issues/27688
  if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

  callbacks=[]
  train_time_callback = TrainTimeCallback()
  batch_summary_callback = BatchAccuracyAndLossSummaryCallback(log_dir, DATASET_SIZE)

  if DISTRIBUTION_STRATEGY_TYPE == 'tf.distribute.experimental.TPUStrategy':
    # epoch and accuracy constants are not supported when training on TPU.
    checkpoints_file_path = checkpoints_dir + "/epochs_tpu.hdf5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_file_path, verbose=1, mode='max')
    callbacks=[tensorboard_callback, checkpoint_callback, train_time_callback]
  else:
    if EMBEDDINGS_MODE == EMBEDDINGS_MODE_TYPE.manual or DISTRIBUTION_STRATEGY_TYPE == 'tf.distribute.experimental.MultiWorkerMirroredStrategy':
      # accuracy fails for adagrad
      # for some reason accuracy is not available for EMBEDDINGS_MODE_TYPE.manual
      # for some reason accuracy is not available for MultiWorkerMirroredStrategy
      checkpoints_file_path = checkpoints_dir + "/epochs:{epoch:03d}.hdf5"
    else:
      checkpoints_file_path = checkpoints_dir + "/epochs:{epoch:03d}-accuracy:{accuracy:.3f}.hdf5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_file_path, verbose=1, mode='max')
    callbacks=[tensorboard_callback, batch_summary_callback, train_time_callback]
    #callbacks=[tensorboard_callback, checkpoint_callback, batch_summary_callback, train_time_callback]
    #callbacks=[tensorboard_callback, checkpoint_callback, train_time_callback]

  verbosity = 1 if TRAIN_LOCATION == TRAIN_LOCATION_TYPE.local else 2
  logging.info('training keras model')
  # training_ds = get_dataset('train').repeat(EPOCHS)
  # model.fit(training_ds, epochs=EPOCHS, verbose=verbosity, callbacks=callbacks, steps_per_epoch = get_steps_per_epoch())
  training_ds = get_dataset('train').repeat(EPOCHS)
  eval_ds = get_dataset('test').repeat(EPOCHS)
  # steps_per_epoch and validation_steps are required for MultiWorkerMirroredStrategy
  model.fit(
    training_ds,
    epochs=EPOCHS,
    verbose=verbosity,
    callbacks=callbacks,
    steps_per_epoch=get_training_steps_per_epoch())
    # validation_data=eval_ds,
    # validation_steps=get_validation_steps_per_epoch())
  #model.fit(training_ds, epochs=EPOCHS, verbose=verbosity, callbacks=callbacks)
  eval_ds = get_dataset('test')
  logging.info("done training keras model, evaluating model")
  loss, accuracy = model.evaluate(eval_ds, verbose=verbosity, steps=get_validation_steps_per_epoch(), callbacks=[tensorboard_callback, batch_summary_callback])
  logging.info("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))
  logging.info(model.summary())
  logging.info("done evaluating keras model")

def train_keras_model_to_estimator(strategy, model, model_dir):
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

def train_keras_functional_no_feature_layer(strategy, model_dir):
  train_and_evaluate_keras_model(create_keras_model_functional_no_feature_layer(), model_dir)

def train_keras_functional_wide_and_deep(strategy, model_dir):
  train_and_evaluate_keras_model(create_keras_model_functional_wide_and_deep(), model_dir)

def train_keras_to_estimator_sequential(strategy, model_dir):
  train_keras_model_to_estimator(strategy, create_keras_model_sequential(), model_dir)

def train_keras_to_estimator_functional(strategy, model_dir):
  train_keras_model_to_estimator(strategy, create_keras_model_functional(), model_dir)

def train_estimator(strategy, model_dir):
  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig(
          train_distribute=strategy,
          eval_distribute=strategy)
  feature_columns = create_feature_columns()
  estimator = tf.estimator.DNNClassifier(
      optimizer=tf.optimizers.SGD(learning_rate=0.05),
      feature_columns=feature_columns,
      hidden_units=[2560, 1024, 256],
      model_dir=model_dir,
      config=config,
      n_classes=2)
  logging.info('training and evaluating estimator model')
  # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
  # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train').repeat(EPOCHS), max_steps=get_max_steps()),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('done evaluating estimator model')

def train_estimator_wide_and_deep(strategy, model_dir):
  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig(
          train_distribute=strategy,
          eval_distribute=strategy)

  linear_feature_columns = create_linear_feature_columns()
  corpus_dict = get_corpus_dict()
  categorical_feature_columns_and_embeddings = create_categorical_embeddings_feature_columns(corpus_dict)
  categorical_feature_columns = list(tf.feature_column.indicator_column(embedding.categorical_column) for embedding in categorical_feature_columns_and_embeddings)
  embeddings = list(embedding for embedding in categorical_feature_columns_and_embeddings)

  crossed_columns = list(tf.feature_column.indicator_column(tf.feature_column.crossed_column(
    keys,
    min(max(len(corpus_dict[keys[0]]), len(corpus_dict[keys[1]])), 100000))) for keys in KEY_CROSSES)

  deep_model_feature_columns = embeddings + linear_feature_columns
  wide_model_feature_columns = categorical_feature_columns + crossed_columns
  estimator = tf.estimator.DNNLinearCombinedClassifier(
      dnn_optimizer=tf.optimizers.SGD(learning_rate=0.05),
      linear_optimizer=tf.optimizers.SGD(learning_rate=0.05),
      linear_feature_columns=wide_model_feature_columns,
      dnn_feature_columns=deep_model_feature_columns,
      dnn_hidden_units=[2560, 1024, 256],
      model_dir=model_dir,
      config=config,
      n_classes=2)
  logging.info('training wide and deep estimator model')
  # Need to specify both max_steps and epochs. Each worker will go through epoch separately.
  # see https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate?version=stable
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train').repeat(EPOCHS), max_steps=get_max_steps()),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('done evaluating wide and deep estimator model')

def train_custom_loop_keras_sequential(strategy, model_dir):
  train_custom_loop(strategy, create_keras_model_sequential(), model_dir)

def train_custom_loop_keras_model_functional_no_feature_layer(strategy, model_dir):
  train_custom_loop(strategy, create_keras_model_functional_no_feature_layer(), model_dir)

def train_custom_loop(strategy, model, model_dir):
  logging.info('training using custom loop')

  log_dir= os.path.join(model_dir, "logs")
  if not os.path.exists(log_dir):
        os.makedirs(log_dir)
  file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
  file_writer.set_as_default()
  # This is fine for MirroredStrategy and TPUStrategy, but has to be changed to run on chief only
  # once multi-node training is supported (ParameterServer, MultiWorkerMirroredStrategy).
  batch_summary_callback = BatchAccuracyAndLossSummaryCallback(DATASET_SIZE)
  train_time_callback = TrainTimeCallback()
  checkpoints_dir= os.path.join(model_dir, "checkpoints")
  if not os.path.exists(checkpoints_dir):
      os.makedirs(checkpoints_dir)

  checkpoints_file_path = checkpoints_dir + "/epochs:{epoch:03d}.hdf5"

  loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
  loss_fn = lambda labels, predictions: tf.reduce_mean(loss_object(labels, predictions))

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  train_accuracy = tf.keras.metrics.BinaryAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.BinaryAccuracy(
      name='test_accuracy')
  optimizer = tf.optimizers.SGD(learning_rate=0.05)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  @tf.function
  def train_step(examples, labels):
    with tf.GradientTape() as tape:
        predictions = model(examples, training=True)
        loss = loss_fn(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_accuracy.update_state(labels, predictions)
    return loss

  @tf.function
  def test_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    test_accuracy.update_state(labels, predictions)
    test_loss.update_state(loss)
    return loss

  train_dist_dataset = get_dataset('train')
  train_time_callback.on_train_begin()
  for epoch in range(EPOCHS):
    batch_summary_callback.on_epoch_begin(epoch, {})
    train_time_callback.on_epoch_begin(epoch, {})

    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for (examples, labels) in train_dist_dataset:
      # batch losses from all replicas
      batch_loss_all_replicas = strategy.experimental_run_v2(train_step, args=(examples, labels))
      # reduced to a single number both across replicas and across the bacth size
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, batch_loss_all_replicas, axis=None)
      total_loss += loss
      train_loss = total_loss / num_batches
      batch_summary_callback.on_train_batch_end(
        num_batches,
        {'accuracy': train_accuracy.result(),
        'loss': train_loss })
      num_batches += 1
    checkpoint.save(checkpoints_file_path.format(epoch=epoch))
    train_time_callback.params = {'steps' : num_batches}
    train_time_callback.on_epoch_end(epoch, {})
    logging.info("Epoch: {} - Loss: {}, Accuracy: {}".format(epoch, loss, train_accuracy.result()))
  train_time_callback.on_train_end()

  test_dist_dataset = get_dataset('test')

  # Keras evaluation
  # logging.info("done training keras model, evaluating model")
  # verbosity = 1 if TRAIN_LOCATION == TRAIN_LOCATION_TYPE.local else 2
  # loss, accuracy = model.evaluate(test_dist_dataset, verbose=verbosity)
  # logging.info("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))

  # TEST LOOP
  for (examples, labels) in test_dist_dataset:
    batch_loss_all_replicas = strategy.experimental_run_v2(test_step, args=(examples, labels))

  logging.info("Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}".format(
    train_loss,
    train_accuracy.result(),
    test_loss.result(),
    test_accuracy.result()))

  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()

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
        type=int)

    args_parser.add_argument(
        '--embeddings-mode',
        help='Embeddings mode.',
        choices=EMBEDDINGS_MODE_TYPE_VALUES,
        default='hashbucket')

    args_parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Ignored by this script.',
        default=False)

    args_parser.add_argument(
        '--ai-platform-mode',
        help='Ignored by this script.',
        default=None)

    args_parser.add_argument(
        '--image-name',
        help='Ignored by this script.',
        default=None)

    args_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disabling GPUs - forces training to happen on CPU.',
        default=False)

    return args_parser.parse_args()

def setup_environment(args):
  global TRAIN_LOCATION
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
    global DISTRIBUTION_STRATEGY_TYPE
    global EMBEDDINGS_MODE
    args = get_args()

    TRAIN_LOCATION = TRAIN_LOCATION_TYPE[args.train_location]
    logging.info('train_location: ' + str(TRAIN_LOCATION))

    if TRAIN_LOCATION != TRAIN_LOCATION_TYPE.local:
      logging_client = google.cloud.logging.Client()
      logging_client.setup_logging()
    else:
      # disable MKL if run locally, see https://b.corp.google.com/issues/149489290
      os.environ['TF_DISABLE_MKL'] = '1'

    logging.getLogger().setLevel(logging.INFO)
    logging.info('>>>>>>>>>>>>>>>>>>> trainer started <<<<<<<<<<<<<<<<<<<<<<<')
    logging.info('trainer called with following arguments:')
    logging.info(' '.join(sys.argv))

    if args.no_gpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
      logging.info('forcing no GPU')

    logging.warning('tf version: ' + tf.version.VERSION)

    # Uncomment to see environment variables
    # logging.warning(os.system('env'))

    # Uncomment this line to see Op device placement
    # tf.debugging.set_log_device_placement(True)

    # https://github.com/tensorflow/tensorflow/issues/34568
    # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model_with_multiworkermirroredstrategy
    # Currently there is a limitation in MultiWorkerMirroredStrategy where TensorFlow ops need to be created after the instance of strategy is created.
    distribution_strategy = None
    # TPU won't work on this sample because strings are not supported by TPU, see:
    # https://cloud.google.com/tpu/docs/troubleshooting#unsupported_data_type
    if args.distribution_strategy == 'tf.distribute.experimental.TPUStrategy':
      tpu = None
      try:
        logging.info('resolving to TPU cluster')
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        logging.info('connecting to TPU cluster')
        tf.config.experimental_connect_to_cluster(tpu)
      except ValueError as e:
        logging.info('error connecting to TPU cluster: %s', e)
        return
      logging.info('initializing TPU system')
      tf.tpu.experimental.initialize_tpu_system(tpu)
      distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
      logging.info('training using TPUStrategy, tpu.cluster_spec: %s', tpu.cluster_spec())
    elif args.distribution_strategy == 'tf.distribute.OneDeviceStrategy':
      if args.no_gpu:
        distribution_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      else:
        distribution_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif args.distribution_strategy:
      distribution_strategy = eval(args.distribution_strategy)()

    logging.info('tensorflow version: ' + tf.version.VERSION)
    logging.info('tensorflow_io version: ' + tf_io.version.VERSION)
    logging.info('device_lib.list_local_devices(): ' + str(device_lib.list_local_devices()))

    DATASET_SOURCE = DATASET_SOURCE_TYPE[args.dataset_source]
    logging.info('dataset_source: ' + str(DATASET_SOURCE))
    DATASET_SIZE = DATASET_SIZE_TYPE[args.dataset_size]
    logging.info('dataset_size: ' + str(DATASET_SIZE))
    DISTRIBUTION_STRATEGY_TYPE = args.distribution_strategy
    logging.info('distribution_strategy: ' + str(DISTRIBUTION_STRATEGY_TYPE))

    model_dir = args.job_dir
    # if TRAIN_LOCATION == TRAIN_LOCATION_TYPE.cloud and os.environ.get('HOSTNAME'):
    #   model_dir = os.path.join(model_dir, os.environ.get('HOSTNAME'))
    # model_dir = os.path.join(model_dir, args.training_function, 'model.joblib')
    logging.info('Model will be saved to "%s..."', model_dir)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    log_dir= os.path.join(model_dir, "logs")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    logging.info('training_function arg: ' + str(args.training_function))
    training_function = getattr(sys.modules[__name__], args.training_function)
    logging.info('training_function: ' + str(training_function))

    EMBEDDINGS_MODE = EMBEDDINGS_MODE_TYPE[args.embeddings_mode]
    if args.training_function == 'train_keras_functional_no_feature_layer':
      EMBEDDINGS_MODE = EMBEDDINGS_MODE_TYPE.manual
    logging.info('embeddings_mode: ' + str(EMBEDDINGS_MODE))

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
