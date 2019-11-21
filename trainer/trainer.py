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

FLAGS = flags.FLAGS
flags.DEFINE_string("job-dir", "", "Job directory")

LOCATION = 'us'
PROJECT_ID = "alekseyv-scalableai-dev"
GOOGLE_APPLICATION_CREDENTIALS = "alekseyv-scalableai-dev-077efe757ef6.json"

# # Download options.
# DATA_URL = 'gs://alekseyv-scalableai-dev-public-bucket/criteo_kaggle.tar.gz'

# DATASET_ID = 'criteo_kaggle'

BATCH_SIZE = 256

CSV_SCHEMA = [
      bigquery.SchemaField("label", "INTEGER", mode='REQUIRED'),
      # using strings because of https://github.com/tensorflow/io/issues/619
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

def create_bigquery_dataset_if_necessary(dataset_id):
  # Construct a full Dataset object to send to the API.
  client = bigquery.Client(project=PROJECT_ID)
  dataset = bigquery.Dataset(bigquery.dataset.DatasetReference(PROJECT_ID, dataset_id))
  dataset.location = LOCATION

  try:
    dataset = client.create_dataset(dataset)  # API request
    return True
  except GoogleAPIError as err:
    if err.code != 409: # http_client.CONFLICT
      raise
  return False

def load_data_into_bigquery(url, dataset_id, table_id):
  create_bigquery_dataset_if_necessary(dataset_id)
  client = bigquery.Client(project=PROJECT_ID)
  dataset_ref = client.dataset(dataset_id)
  table_ref = dataset_ref.table(table_id)
  job_config = bigquery.LoadJobConfig()
  job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
  job_config.source_format = bigquery.SourceFormat.CSV
  #job_config.autodetect = True
  job_config.schema = CSV_SCHEMA

  load_job = client.load_table_from_uri(
      url, table_ref, job_config=job_config
  )
  print("Starting job {}".format(load_job.job_id))

  load_job.result()  # Waits for table load to complete.
  print("Job finished.")

  destination_table = client.get_table(table_ref)
  print("Loaded {} rows.".format(destination_table.num_rows))

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
  #print(query_job.result())
  #print(query_job.errors)
  #print(df)

  mean_dict = dict((field[0].replace('avg_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('avg'))
  std_dict = dict((field[0].replace('std_', ''), df[field[0]][0]) for field in df.items() if field[0].startswith('std'))
  return (mean_dict, std_dict)

def transofrom_row(row_dict, mean_dict, std_dict):
  dict_without_label = row_dict.copy()
  #tf.print(dict_without_label)
  label = dict_without_label.pop('label')
  for field in CSV_SCHEMA:
    if (field.name.startswith('int')):
        if dict_without_label[field.name] == 0:
            value = float(dict_without_label[field.name])
            dict_without_label[field.name] = (value - mean_dict[field.name]) / std_dict[field.name]
        else:
            dict_without_label[field.name] = 0.0 # don't use normalized 0 value for nulls
  return (dict_without_label, label)

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

  dataset = read_session.parallel_read_rows()
  transformed_ds = dataset.map (lambda row: transofrom_row(row, mean_dict, std_dict))
  return transformed_ds

def get_vocabulary_size_dict():
  client = bigquery.Client(location="US", project=PROJECT_ID)
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
      location="US",
  )  # API request - starts the query

  df = query_job.to_dataframe()
  #print(query_job.result())
  #print(query_job.errors)
  #print(df)
  dictionary = dict((field[0], df[field[0]][0]) for field in df.items())
  #print(dir(df))
  return dictionary

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

def create_feature_columns(categorical_vocabulary_size_dict):
  feature_columns = []
  feature_columns.extend(list(tf.feature_column.numeric_column(field.name, dtype=tf.dtypes.float32)  for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label'))
  feature_columns.extend(list(create_categorical_feature_column(categorical_vocabulary_size_dict, key) for key, _ in categorical_vocabulary_size_dict.items()))
  return feature_columns

def main(argv):
    if len(argv) < 1:
      raise app.UsageError("Too few command-line arguments.")

    model_dir = os.path.join(sys.argv[1], 'model.joblib')
    logging.info('Model will be saved to "%s..."', model_dir)

    #tf.debugging.set_log_device_placement(True)
    print("tf.config.experimental.list_logical_devices(GPU): " + str(tf.config.experimental.list_logical_devices('GPU')))
    print("tf.config.experimental.list_physical_devices(GPU): " + str(tf.config.experimental.list_physical_devices('GPU')))
    print("device_lib.list_local_devices(): " + str(device_lib.list_local_devices()))
    print("tf.test.is_gpu_available(): " + str(tf.test.is_gpu_available()))

    print("reading categorical_vocabulary_size_dict")
    categorical_vocabulary_size_dict = get_vocabulary_size_dict()

    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
      feature_columns = create_feature_columns(categorical_vocabulary_size_dict)
      print("categorical_vocabulary_size_dict: " + str(categorical_vocabulary_size_dict))
      feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
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

      #training_ds = read_bigquery('criteo_kaggle','days_strings').take(1000000).shuffle(10000).batch(BATCH_SIZE).prefetch(100)
      #training_ds = read_bigquery('criteo_kaggle','days').skip(100000).take(50000).shuffle(10000).batch(BATCH_SIZE)
      training_ds = read_bigquery('criteo_kaggle','days').take(1000000).shuffle(10000).batch(BATCH_SIZE)
      print('checking dataset')
      # row_index = 0
      # for row in training_ds.take(2):
      #     print(">>>>>> row %d: %s" % (row_index, row))
      #     row_index += 1

      if not os.path.exists(model_dir + "/checkpoints"):
          os.makedirs(model_dir + "/checkpoints")

      log_dir= model_dir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, embeddings_freq=0)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1, profile_batch=0)

      #filepath= model_dir + "/checkpoints/epochs:{epoch:03d}-accuracy:{accuracy:.3f}.hdf5"
      #checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, mode='max')
      filepath= model_dir + "/checkpoints/epochs:{epoch:03d}.hdf5"
      checkpoint = ModelCheckpoint(filepath, verbose=1, mode='max')

      model.fit(training_ds, epochs=5, verbose=2,
      callbacks=[tensorboard_callback, checkpoint]
      )
    #, callbacks=[tensorboard_callback, checkpoint]
    print('evaluating model')

    eval_ds = read_bigquery('criteo_kaggle','days').skip(100000).take(50 * BATCH_SIZE).batch(BATCH_SIZE)
    # row_index = 0
    # for row in eval_ds.take(2):
    #     print(">>>>>> row %d: %s" % (row_index, row))
    #     row_index += 1

    loss, accuracy = model.evaluate(eval_ds)
    print("Eval - Loss: {}, Accuracy: {}".format(loss, accuracy))


if __name__ == '__main__':
  print('executable')
  print(sys.executable)
  print(sys.version)
  print(sys.version_info)

  #print('pip')
  #print(os.system('pip --version'))
  #print(os.system('pip list'))

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
  os.environ['PROJECT_ID'] = PROJECT_ID
  print(os.system('pwd'))
  print(os.system('ls -al'))
  if (os.environ.get('CLOUDSDK_METRICS_COMMAND_NAME') == 'gcloud.ai-platform.local.train'):
    print('training locally')
    print('removing TF_CONFIG')
    os.environ.pop('TF_CONFIG')
  else:
    print('training in cloud')
    os.system('gsutil cp gs://alekseyv-scalableai-dev-private-bucket/criteo/alekseyv-scalableai-dev-077efe757ef6.json .')
    os.environ[ "GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + '/' + GOOGLE_APPLICATION_CREDENTIALS
  print(os.environ)
  print(os.system('cat ${GOOGLE_APPLICATION_CREDENTIALS}'))
  app.run(main)

