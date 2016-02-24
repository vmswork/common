"""
Compute final outputs of a tensorflow network from kaldi features.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
import tf_model
import os
import kaldi_helpers
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
if len(FLAGS.__flags) == 0:
  flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
  flags.DEFINE_string('features', './t.ark',
                        'Input features file (must be written with ark,t).')
  flags.DEFINE_string('gpu', '1',
                        'GPU id.')
  flags.DEFINE_string('occupances', './final.occs',
                        'File with counts in kaldi format to do post2like.')
  flags.DEFINE_string('train_dir', '.',
                        '.')                        
  flags.DEFINE_integer('transpose_input', 1,
                        'If 1 rearanging incoming features from channels_numberXvectors_in_channel to vectors_in_channelsXchannel_number, otherwise do nothing.')
                        
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

VALIDATION_FILE = 'train_1280.tfrecords'
FEATURE_DIMENSIONALITY = 1320
FEATURE_INPUT_SHAPE = [11, 3, 40]
# FEATURE_OUTPUT_SHAPE = [11, 40, 3]

# NUM_VAL_SAMPLES = 498688
NUM_VAL_SAMPLES = 1280
VAL_BATCH_SIZE = 652

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
  'vector': tf.FixedLenFeature([], tf.string),
  'label': tf.FixedLenFeature([], tf.int64),
  })  
  
  
  
  # features = tf.parse_single_example(serialized_example, dense_keys=['vector', 'label'], dense_types=[tf.string, tf.int64])
  # Convert from a scalar string tensor (whose single string has
  # length tf_model.IMAGE_PIXELS) to a uint8 tensor with shape
  # [tf_model.IMAGE_PIXELS].
  image = tf.decode_raw(features['vector'], tf.float32)
  image.set_shape([FEATURE_DIMENSIONALITY])
  if FLAGS.transpose_input:
    image = tf.reshape(image, FEATURE_INPUT_SHAPE)
    image = tf.transpose(image, perm=[0,2,1])
    image = tf.reshape(image, [-1])

  # print("Image shape is %s" %(image.shape))
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.
  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)
  return image, label
def eval_inputs(batch_size=VAL_BATCH_SIZE, file=VALIDATION_FILE):
  filename = os.path.join(FLAGS.train_dir,
                          file)
  with tf.variable_scope("input", reuse=True):
    filename_queue = tf.train.string_input_producer(
        [filename])
    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    images, sparse_labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=2)
    return images, sparse_labels
  

def produce_likelihoods():
  with kaldi_helpers.kaldi_data('./t.ark') as kd:
    batch1 = kd.read_utterance(-1)
    u1 = batch1.next()
    u2 = batch1.next()
  print(np.shape(u1[1]))
  with kaldi_helpers.kaldi_data(FLAGS.occupances) as kd:
    logprioirs = kd.read_counts()
  
  with tf.Graph().as_default():
    val_images, val_labels = eval_inputs()
    images = tf.placeholder(tf.float32, shape=(None, 1320))
    labels = tf.placeholder(tf.int32, shape=(None))
    logits = tf_model.inference(images,
                               2048,
                               2048,
                               2048) 
    loss = tf_model.loss(logits, labels)                                
    saver = tf.train.Saver()
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    saver.restore(sess, '../../../data/tf_fbank_deltas_nocmvn/cnnmodel-31530')
    vi, vl = sess.run([val_images, val_labels])
    r = sess.run([logits], feed_dict={images: vi})
    l = r[0] - logprioirs
    with kaldi_helpers.kaldi_data('./t_like_u1.ark', 'w') as kd:
      kd.write_utterance([[u1[0], l]])
    r = sess.run([loss], feed_dict={images: vi, labels: vl})
    print(r)
    r = sess.run([loss], feed_dict={images: u1[1], labels: vl})
    print(r)
    l = vi
    # with kaldi_helpers.kaldi_data('./t_feats_tfr.ark', 'w') as kd:
    #  kd.write_utterance([[u1[0], l]])    
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)    
    sess.close()
    

def main(_):
  produce_likelihoods()
  
if __name__ == '__main__':
  tf.app.run()