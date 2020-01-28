#!/usr/bin/env python
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

from setuptools import find_packages
from setuptools import setup

# Note: it is important to specify pandas version, otherwise pandas0.17.1 is installed by default
#    'pandas==0.24.2',
#  --no-deps is required for tensorflow-io because of https://github.com/tensorflow/io/issues/124
# tf-nightly==2.1.0.dev20191125 vs tensorflow==2.0.1 vs tf-nightly

REQUIRED_PACKAGES = [
    'pandas==0.24.2',
    'tensorflow==2.0.1',
    'tensorflow-io==0.10.0',
    'google-cloud-bigquery==1.22.0',
    'google-cloud-bigquery-storage==0.7.0',
    'google-cloud-logging==1.14.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    #dependency_links=['https://storage.googleapis.com/alekseyv-scalableai-dev-public-bucket/temp/tensorflow_io-0.10.0-cp27-cp27mu-manylinux2010_x86_64.whl'],
    packages=find_packages(),
    include_package_data=True,
    description='Criteo dataset trainer',
    # package_data = {
    #     # If any package contains *.txt or *.rst files, include them:
    #     '': ['*.json'],
    # },
)