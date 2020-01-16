#!/usr/bin/env python

from __future__ import print_function

import jinja2
import sys
import argparse
import os
from datetime import datetime


BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
DISTRIBUTION_STRATEGY_TYPE_VALUES = 'tf.distribute.MirroredStrategy tf.distribute.experimental.ParameterServerStrategy ' \
  'tf.distribute.experimental.MultiWorkerMirroredStrategy tf.distribute.experimental.CentralStorageStrategy'

CURRENT_DATE=datetime.now().strftime('date_%Y%m%d_%H%M%S')
MODEL_NAME=CURRENT_DATE
MODEL_DIR='gs://{}/{}/model'.format(BUCKET_NAME, MODEL_NAME)

args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    '--model-name',
    help='model name, not used.')

args_parser.add_argument(
    '--job-dir',
    help='folder or GCS location to write checkpoints and export models.')

args_parser.add_argument(
    '--distribution-strategy',
    help='Distribution strategy to use.',
    choices=DISTRIBUTION_STRATEGY_TYPE_VALUES.split(' '))

args, unknown = args_parser.parse_known_args()

ps_replicas=0
worker_replicas=1
num_gpus_per_worker=1

if args.distribution_strategy == "tf.distribute.MirroredStrategy" or args.distribution_strategy == "tf.distribute.experimental.CentralStorageStrategy":
    num_gpus_per_worker=2
    worker_replicas=1
elif args.distribution_strategy == "tf.distribute.experimental.ParameterServerStrategy":
    ps_replicas=1
    worker_replicas=2
    num_gpus_per_worker=2 # ???
elif args.distribution_strategy == "tf.distribute.experimental.MultiWorkerMirroredStrategy":
    worker_replicas=2
    num_gpus_per_worker=2

trainer_cmd_args = ' '.join(["--job-dir=" + MODEL_DIR, "--train-location=cloud"] + sys.argv[1:])

with open(os.path.dirname(os.path.realpath(__file__)) + "/template.yaml.jinja", "r") as f:
  print(jinja2.Template(f.read()).render(
      ps_replicas=ps_replicas,
      worker_replicas=worker_replicas,
      num_gpus_per_worker=num_gpus_per_worker,
      train_dir=MODEL_DIR,
      cmdline_args=trainer_cmd_args
      ))