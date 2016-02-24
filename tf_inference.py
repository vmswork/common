"""
Compute final outputs of a tensorflow network from kaldi features.

TODO: get rid of process_data in tf_model for numpy arrays to leave
      one processing method operating on tensors


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os.path
import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
# import tf_model
import os
from kaldi_helpers import kaldi_data
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
if len(FLAGS.__flags) == 0:
  flags.DEFINE_string('tf_model_path', 'tf_model', 'Path where model file is stored.')
  flags.DEFINE_string('tf_ckpt_path', 'cnnmodel-31530', 'Path where checkpoint file is stored.')
  flags.DEFINE_integer('batch_size', 256, 'Batch size.')
  flags.DEFINE_string('features_rspec', './local/common/t.ark',
                        'Input features file (must be written with ark,t). If \
                            ark,t:- then read from stdin.')
  flags.DEFINE_string('prob_wspec', 'w.ark',
                        'Where to write probabilities.')
  flags.DEFINE_integer('post2like', 1,
                        'If 1 then convert posteriors to likelihoods.')                        
  flags.DEFINE_string('gpu', '1',
                        'GPU id.')
  flags.DEFINE_string('occupances', 'final.occs',
                        'File with counts in kaldi format to do post2like.')
  flags.DEFINE_string('data_dir', 'local/common/tf_decode/',
                        'This string is added to tf_model_path, tf_ckpt_path, occupances.')
  flags.DEFINE_integer('transpose_input', 1,
                        'If 1 rearanging incoming features from channels_numberXvectors_in_channel to vectors_in_channelsXchannel_number, otherwise do nothing.')
                        
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def import_tf_model():
  model_path = os.path.join(FLAGS.data_dir,
                          FLAGS.tf_model_path)
  model_dir = os.path.dirname(model_path)
  madel_name = os.path.basename(model_path)
  sys.path.insert(0, model_dir)
  tf_model = __import__(madel_name)
  # print("%s from %s is loaded" % (madel_name, model_dir))
  return tf_model

def get_logpriors():
  logpriors = 0
  if FLAGS.post2like:
    with kaldi_data(os.path.join(FLAGS.data_dir,
                            FLAGS.occupances)) as kd:
      logpriors = kd.read_counts()
  return logpriors
  
def produce_likelihoods():
  tf_model = import_tf_model()
  logpriors = get_logpriors()
  with tf.Graph().as_default():
    features = tf.placeholder(tf.float32, shape=(None, None))
    logits = tf_model.inference(features) 
    ckpt_path =  os.path.join(FLAGS.data_dir,
                          FLAGS.tf_ckpt_path)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, ckpt_path)
    with kaldi_data(FLAGS.features_rspec) as kd_reader:
      with kaldi_data(FLAGS.prob_wspec, 'w') as kd_writer:
        for d in kd_reader.read_utterance(FLAGS.batch_size):
          utterance_id = d[0]
          batch = tf_model.process_data(d[1], FLAGS.transpose_input)
          r = sess.run([logits], feed_dict={features: batch})
          kd_writer.write_batches([utterance_id, r[0] - logpriors])
    sess.close()
  return
  
      

def main(_):
  produce_likelihoods()
  
if __name__ == '__main__':
  tf.app.run()