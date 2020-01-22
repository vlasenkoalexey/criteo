#!/usr/bin/env python

from __future__ import print_function

import jinja2
import sys
import argparse
import os
from datetime import datetime


BUCKET_NAME="alekseyv-scalableai-dev-criteo-model-bucket"
DISTRIBUTION_STRATEGY_TYPE_VALUES = 'tf.distribute.MirroredStrategy tf.distribute.experimental.ParameterServerStrategy ' \
  'tf.distribute.experimental.MultiWorkerMirroredStrategy tf.distribute.experimental.CentralStorageStrategy ' \
  'tf.distribute.experimental.TPUStrategy'

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

num_ps=0
num_workers=0
num_gpus_per_worker=0
num_tpus=0

if args.distribution_strategy == "tf.distribute.MirroredStrategy":
    num_gpus_per_worker=2
    num_workers=0 # chief only
elif args.distribution_strategy == "tf.distribute.experimental.CentralStorageStrategy":
    num_ps=1 # see https://b.corp.google.com/issues/148108526 why PS is necessary
    num_workers=0
    num_gpus_per_worker=0
elif args.distribution_strategy == "tf.distribute.experimental.ParameterServerStrategy":
    num_ps=1
    num_workers=2
    num_gpus_per_worker=2 # ???
elif args.distribution_strategy == "tf.distribute.experimental.MultiWorkerMirroredStrategy":
    num_workers=2
    num_gpus_per_worker=2
elif args.distribution_strategy == "tf.distribute.experimental.TPUStrategy":
    num_workers=0
    num_gpus_per_worker=0
    num_tpus=32 # minimal available number for central1-a

trainer_cmd_args = ' '.join(["--train-location=cloud"] + sys.argv[1:])

with open(os.path.dirname(os.path.realpath(__file__)) + "/template.yaml.jinja", "r") as f:
  print(jinja2.Template(f.read()).render(
      num_ps=num_ps,
      num_workers=num_workers,
      num_gpus_per_worker=num_gpus_per_worker,
      num_tpus=num_tpus,
      train_dir=args.job_dir,
      cmdline_args=trainer_cmd_args
      ))
