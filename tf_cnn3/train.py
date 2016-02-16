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
"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See tensorflow/g3doc/how_tos/reading_data.md#reading-from-files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
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
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
if len(FLAGS.__flags) == 0:
  flags.DEFINE_float('learning_rate', 0.08, 'Initial learning rate.')
  flags.DEFINE_integer('num_epochs', 25, 'Number of epochs to run trainer.')
  flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
  flags.DEFINE_integer('hidden2', 2048, 'Number of units in hidden layer 2.')
  flags.DEFINE_integer('hidden3', 2048, 'Number of units in hidden layer 3.')
  flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
  flags.DEFINE_string('train_dir', '.',
                        'Directory with the training data.')
  flags.DEFINE_string('gpu', '1',
                        'GPU id.')
  flags.DEFINE_string('transpose_input', '0',
                        'If 1 rearanging incoming features from channels_numberXvectors_in_channel to vectors_in_channelsXchannel_number, otherwise do nothing.')
                        
                        
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
                        
# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train_6457344.tfrecords'
# VALIDATION_FILE = 'dev_498688.tfrecords'
VALIDATION_FILE = 'dev_2012928.tfrecords'
FEATURE_DIMENSIONALITY = 1320
FEATURE_INPUT_SHAPE = [11, 3, 40]
# FEATURE_OUTPUT_SHAPE = [11, 40, 3]

# NUM_VAL_SAMPLES = 498688
NUM_VAL_SAMPLES = 2012928
NUM_TRAIN_SAMPLES = 6457344
VAL_BATCH_SIZE = 256

# TRAIN_FILE = 'train_5120.tfrecords'
# VALIDATION_FILE = 'train_5120.tfrecords'
# NUM_VAL_SAMPLES = 5120
# NUM_TRAIN_SAMPLES = NUM_VAL_SAMPLES



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
def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, tf_model.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, tf_model.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE)
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)
    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=100000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=100000)
    return images, sparse_labels
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
def update_lrs(lrs, epochs_loss, epochs_fac):
  print(np.abs(epochs_loss[-1] - epochs_loss[-2]))
  if np.abs(epochs_loss[-1] - epochs_loss[-2]) < 0.04:
     lrs.append(lrs[-1] / 2)
     print("LR is %.10f" %(lrs[-1]))
def run_training():
  """Train MNIST for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # with tf.variable_scope('training') as scope:
    # Input images and labels.
    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs)
    # Eval inputs
    val_images, val_labels = eval_inputs()
    # Build a Graph that computes predictions from the inference model.
    logits = tf_model.inference(images,
                             FLAGS.hidden1,
                             FLAGS.hidden2,
                             FLAGS.hidden3)
    frame_accuracy = tf_model.evaluation(logits, labels)
    # Add to the Graph the loss calculation.
    loss = tf_model.loss(logits, labels)
    evaluation = tf_model.evaluation(logits, labels)
    ce_summ = tf.scalar_summary("cross entropy", loss)
    # with tf.variable_scope("hidden1", reuse=True):
    # weights_summ_h1=tf.histogram_summary("h1", weights))  
    lr = tf.Variable(float(FLAGS.learning_rate), name='lr')
    # Add to the Graph operations that train the model.
    train_op = tf_model.training(loss, lr)
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    # The op for initializing the variables.
    init_op = tf.initialize_all_variables()
    # Create a session for running operations in the Graph.
    sess = tf.Session()
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter("./data/", sess.graph_def)
    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print(sess.run([images, labels]))
    try:
      epochs_loss = [1000]
      epochs_fac = [0]
      lrs = [float(FLAGS.learning_rate)]
      step = 0
      iter = 0
      train_batches = int(NUM_TRAIN_SAMPLES / FLAGS.batch_size)
      val_batches = int(NUM_VAL_SAMPLES / VAL_BATCH_SIZE)
      # summary_writer.add_graph(sess.graph_def)
      # tf.train.write_graph(sess.graph_def, './data_g/','graph.pbtxt')      
      while not coord.should_stop():
        start_time = time.time()
        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        _, loss_value, fac_value = sess.run([train_op, loss, evaluation],
                feed_dict={lr: lrs[-1]})
        duration = time.time() - start_time
        # Print an overview fairly often.
        if step % 100 == 0:
          summary_str = sess.run([summary_op, loss])
          summary_writer.add_summary(summary_str[0], step)          
          print('Step %d: loss = %.2f (%.3f sec), fac = %.2f' % 
              (step, loss_value, duration, fac_value / FLAGS.batch_size))
        if step % int(NUM_TRAIN_SAMPLES / FLAGS.batch_size) == 0:
          saver.save(sess, FLAGS.train_dir + '/cnnmodel', global_step=step)
          it_loss = it_fac = 0
          print('Validating...')
          for i in range(val_batches):
            if i % 100 == 0: print('batch: %d' % (i))
            vi, vl = sess.run([val_images, val_labels])
            # print(vl.mean())
            loss_val_value, fac_value = sess.run([loss, frame_accuracy], feed_dict={images: vi, labels: vl})
            it_loss += loss_val_value
            it_fac += fac_value / VAL_BATCH_SIZE
          epoch_loss = it_loss / val_batches
          epochs_loss.append(epoch_loss)
          epoch_fac = it_fac / val_batches
          epochs_fac.append(epoch_fac)
          print('Iter %d: cv_loss = %.2f, cv_fac = %.2f' % (iter, epoch_loss, epoch_fac ))
          update_lrs(lrs, epochs_loss, epochs_fac)
          iter += 1
          print(epochs_loss)
          print(lrs)
        
        step += 1
       
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
def main(_):
  run_training()
if __name__ == '__main__':
  tf.app.run()