
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import inspect

import reader
import os
import numpy as np


def data_type():
  return tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps

    #self.epoch_size = len(data)
    self.epoch_size = len(data)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)


    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, tf.float32)
    self._input_data = tf.placeholder(tf.int32,[config.batch_size,config.num_steps])
    self._targets = tf.placeholder(tf.int32,[config.batch_size,config.num_steps])
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype = data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(inputs, num_steps, axis = 1)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property 
  def input_data(self):
      return self._input_data
  @property 
  def targets(self):
      return self._targets

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op




class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 2
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000



def run_epoch(session, model, data, eval_op=None):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  
  for step in range(model.input.epoch_size):


    feed_dict = {model.input_data: data[0][step], model.targets: data[1][step]}

    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  return MediumConfig()


def main(_):

  raw_data = reader.ptb_raw_data()
  true_data, false_data, _ = raw_data
  #true_data, _ = raw_data


  config = get_config()

  true_data, true_data_targets = reader.getinputoutput(true_data, config.batch_size, config.num_steps)
  false_data, false_data_targets = reader.getinputoutput(false_data, config.batch_size, config.num_steps)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("True"):
      true_input = PTBInput(config=config, data=true_data, name="TrainInput")
      with tf.variable_scope("TrueModel", reuse=None, initializer=initializer):
        truem = PTBModel(is_training=True, config=config, input_=true_input)
    with tf.name_scope("Corrupt"):
      corrupt_input = PTBInput(config=config, data=false_data, name="TrainInput")
      with tf.variable_scope("CorruptModel", reuse=None, initializer=initializer):
        corruptm = PTBModel(is_training=True, config=config, input_=corrupt_input)



    sv = tf.train.Supervisor(logdir="home/supersteve/git/nlpchallege/traindir")
    #sv = tf.train.Supervisor()
    with sv.managed_session() as session:
      sv.saver.restore(session, tf.train.latest_checkpoint("home/supersteve/git/nlpchallege/traindir"))
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        truem.assign_lr(session, config.learning_rate * lr_decay)
        corruptm.assign_lr(session, config.learning_rate * lr_decay)


        train_perplexity = run_epoch(session, truem, (true_data, true_data_targets),eval_op=truem.train_op)
        false_train_perplexity = run_epoch(session, corruptm, (false_data, false_data_targets),eval_op=corruptm.train_op)

        print("true"+str(train_perplexity))
        print("false"+str(false_train_perplexity))
        sv.saver.save(session,"home/supersteve/git/nlpchallege/traindir")
        print("saved")
      #sv.saver.save(session,"home/supersteve/git/nlpchallege/traindir")

if __name__ == "__main__":
  tf.app.run()