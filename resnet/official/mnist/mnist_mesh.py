# coding=utf-8
# Copyright 2019 The Mesh TensorFlow Authors.
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

"""MNIST using Mesh TensorFlow and TF Estimator.

This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mnist_dataset as dataset  # local file import
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time as time

tf.flags.DEFINE_string("data_dir", "mnist_data",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", "model_dir", "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 200,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
tf.flags.DEFINE_integer("train_epochs", 40, "Total number of training epochs.")
tf.flags.DEFINE_integer("epochs_between_evals", 1,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
#parse wrong?
#tf.flags.DEFINE_string("mesh_shape", "rows:1;cols:6", "mesh shape")
#tf.flags.DEFINE_string("layout", "filters1:rows;filters2:cols2;filters3:rows;filters4:cols2;filters5:rows;filters6:cols2;",
#                       "layout rules")
tf.flags.DEFINE_integer("num_filters", 64 * 6,
                        "number of filters in a conv layer")

FLAGS = tf.flags.FLAGS


def mnist_model(image, labels, mesh):
  """The model.

  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh

  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  rows_dim = mtf.Dimension("rows_size", 28)
  cols_dim = mtf.Dimension("cols_size", 28)

  classes_dim = mtf.Dimension("classes", 10)
  one_channel_dim = mtf.Dimension("one_channel", 1)

  x = mtf.import_tf_tensor(
      mesh, tf.reshape(image, [FLAGS.batch_size, 28, 28, 1]),
      mtf.Shape(
          [batch_dim, rows_dim, cols_dim, one_channel_dim]))

  fh_dim = mtf.Dimension("fh", 3)
  fw_dim = mtf.Dimension("fw", 3)
  filters1_dim = mtf.Dimension("filters1", FLAGS.num_filters)
  filters2_dim = mtf.Dimension("filters2", FLAGS.num_filters)
  filters3_dim = mtf.Dimension("filters3", FLAGS.num_filters)
  filters4_dim = mtf.Dimension("filters4", FLAGS.num_filters)
  filters5_dim = mtf.Dimension("filters5", FLAGS.num_filters)
  filters6_dim = mtf.Dimension("filters6", FLAGS.num_filters)

  kernel1 = mtf.get_variable(
      mesh, "kernel1", [fh_dim, fw_dim, one_channel_dim, filters1_dim])
  kernel2 = mtf.get_variable(
      mesh, "kernel2", [fh_dim, fw_dim, filters1_dim, filters2_dim])
  kernel3 = mtf.get_variable(
     mesh, "kernel3", [fh_dim, fw_dim, filters2_dim, filters3_dim])
  kernel4 = mtf.get_variable(
     mesh, "kernel4", [fh_dim, fw_dim, filters3_dim, filters4_dim])
  kernel5 = mtf.get_variable(
     mesh, "kernel5", [fh_dim, fw_dim, filters4_dim, filters5_dim])
  kernel6 = mtf.get_variable(
     mesh, "kernel6", [fh_dim, fw_dim, filters5_dim, filters6_dim])

  x = mtf.relu(mtf.conv2d( x, kernel1, strides=[1,1,1,1], padding="SAME"))
  x = mtf.relu(mtf.conv2d( x, kernel2, strides=[1,1,1,1], padding="SAME"))
  x = mtf.relu(mtf.conv2d( x, kernel3, strides=[1,1,1,1], padding="SAME"))
  x = mtf.relu(mtf.conv2d( x, kernel4, strides=[1,1,1,1], padding="SAME"))
  x = mtf.relu(mtf.conv2d( x, kernel5, strides=[1,1,1,1], padding="SAME"))
  x = mtf.relu(mtf.conv2d( x, kernel6, strides=[1,1,1,1], padding="SAME"))
  x = mtf.reduce_mean(x, reduced_dim=filters6_dim)

  # add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1", FLAGS.hidden_size)
  hidden_dim2 = mtf.Dimension("hidden2", FLAGS.hidden_size)
  logits = mtf.Dimension("logits", 10)
  h1 = mtf.layers.dense(
      x, hidden_dim1,
      reduced_dims=x.shape.dims[-2:],
      activation=mtf.relu, name="hidden1")
  h2 = mtf.layers.dense(
      h1, hidden_dim2,
      activation=mtf.relu, name="hidden2")
  logits = mtf.layers.dense(h2, classes_dim, name="logits")
  if labels is None:
    loss = None
  else:
    labels = mtf.import_tf_tensor(
        mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, mtf.one_hot(labels, classes_dim), classes_dim)
    loss = mtf.reduce_mean(loss)
  return logits, loss

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  tf.logging.info("features = %s labels = %s mode = %s params=%s" %
                  (features, labels, mode, params))
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  logits, loss = mnist_model(features, labels, mesh)
  mesh_shape = [("gpu_rows", 1),("gpu_cols", 6)]
  #dependency on ortools, which won't build on power
  #import mesh_tensorflow.auto_mtf
  #layout_rules = mtf.auto_mtf.layout(graph, mesh_shape,[logits,loss])
  layout_rules = [("filters1", "gpu_rows"), ("filters2", "gpu_cols"),
                  ("filters3", "gpu_rows"), ("filters4", "gpu_cols"),
                  ("filters5", "gpu_rows"), ("filters6", "gpu_cols")
                 ]
  mesh_devices = ["gpu:%d" % itm for itm in range(6)]
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    tf_grads = [lowering.export_to_tf_tensor(grad) for grad in var_grads]
    tf_avg_grads = [hvd.allreduce(grad) for grad in tf_grads]
    var_avg_grads = [mtf.import_tf_tensor(mesh, tf_avg_grad, shape=grad.shape) 
                     for tf_avg_grad, grad in zip(tf_avg_grads, var_grads)]
    update_ops = optimizer.apply_grads(var_avg_grads, graph.trainable_variables)
  with tf.variable_scope('', reuse=tf.AUTO_REUSE) as _ :
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  restore_hook = mtf.MtfRestoreHook(lowering)
  tf_logits = lowering.export_to_tf_tensor(logits)
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf.summary.scalar("loss", tf_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)
    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=2,
        defer_build=False, save_relative_paths=True)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    saver_listener = mtf.MtfCheckpointSaverListener(lowering)
    saver_hook = tf.train.CheckpointSaverHook(
        "/tmp/%s"%FLAGS.model_dir,
        save_steps=1000,
        saver=saver,
        listeners=[saver_listener])

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(tf_logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(tf_loss, "cross_entropy")
    tf.identity(accuracy[1], name="train_accuracy")

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar("train_accuracy", accuracy[1])

    # restore_hook must come before saver_hook
    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
        training_chief_hooks=[restore_hook, saver_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(tf_logits, axis=1),
        "probabilities": tf.nn.softmax(tf_logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[restore_hook],
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf_loss,
        evaluation_hooks=[restore_hook],
        eval_metric_ops={
            "accuracy":
            tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
        })


def run_mnist():

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = '0,1,2,3,4,5'
  
  #without specify saverhook  
  #default tf.train.CheckpointSaverHook will be invoked
  #but it won't work with mesh 
  #model_dir = FLAGS.model_dir if hvd.rank() == 0 else None 

  """Run MNIST training and eval loop."""
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir="/tmp/%s"%FLAGS.model_dir,
      config=tf.estimator.RunConfig(session_config=config))

  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(FLAGS.data_dir)
    ds_batched = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds_batched.repeat(FLAGS.epochs_between_evals)
    return ds

  def eval_input_fn():
    return dataset.test(FLAGS.data_dir).batch(
        FLAGS.batch_size).make_one_shot_iterator().get_next()

  # Train and evaluate model.
  for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    mnist_classifier.train(input_fn=train_input_fn, hooks=[bcast_hook])
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("\nEvaluation results:\n\t%s\n" % eval_results)


def main(_):
  hvd.init()
  run_mnist()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
