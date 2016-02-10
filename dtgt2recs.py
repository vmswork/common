# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from  dtgt2numpy import dtgtReader
import sys


flags = tf.app.flags
FLAGS = flags.FLAGS
if len(FLAGS.__flags) == 0:
  tf.app.flags.DEFINE_string('directory', '.',
                             'Directory to download data files and write the '
                             'converted result')
  tf.app.flags.DEFINE_string('features', 'train.dat',
                             'Dat file with features ')
  tf.app.flags.DEFINE_string('labels', 'train.dat',
                             'Tgt file with labels ')
  tf.app.flags.DEFINE_string('setname', 'train',
                             'Set name (train, dev, etc.) ')

FLAGS = tf.app.flags.FLAGS
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to(features, labels, name, mult=1):
  num_examples = labels._header['numberOfFrames']
  if features._header['numberOfFrames'] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  num_to_convert = int(np.floor(num_examples / mult) * mult)
  filename = os.path.join(FLAGS.directory, 
                  ''.join([name,'_' ,str(num_to_convert),'.tfrecords']))
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  features_reader = features.readData(1)
  labels_reader = labels.readData(1)
  for _ in range(num_to_convert):
  # for _ in range(5120):
    feature_vector = features_reader.next()
    label = labels_reader.next()
    feature_vector_raw = feature_vector.astype('float32').tostring()
    # print(feature_vector.mean(), feature_vector.shape)
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(label)),
        'vector': _bytes_feature(feature_vector_raw)}))
    writer.write(example.SerializeToString())
def main(argv):
  features_file_name = FLAGS.features
  labels_file_name = FLAGS.labels
  setname = FLAGS.setname
  features = dtgtReader(features_file_name)
  labels = dtgtReader(labels_file_name)
  convert_to(features, labels, setname, 256)

if __name__ == '__main__':
  tf.app.run()