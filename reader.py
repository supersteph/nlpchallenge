from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import io
import codecs

import numpy as np
#import tensorflow as tf


def _read_words(filename):

  with open(filename, "r", encoding = "utf-8", errors = 'ignore') as f:
    a = f.read()
    a = a.replace("\n", " <eos> \n").replace("\t", " <eos> \n")
    b = a.split("\n")
    return [word for sent in b[::2] for word in sent.split()], [word for sent in b[1::2] for word in sent.split()]


def _build_vocab(filename):
  data, other = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  words = words[:9999]
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  eng, no = _read_words(filename)
  return [word_to_id[word] if word in word_to_id else 9999 for word in eng], [word_to_id[word] if word in word_to_id else 9999 for word in no]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = "ptb.train.txt"

  word_to_id = _build_vocab(train_path)
  correct_data, wrong_data = _file_to_word_ids(train_path, word_to_id)
  vocabulary = len(word_to_id)
  return correct_data, wrong_data, vocabulary

def getinputoutput(raw_data, batch_size, num_steps):
  data = np.asarray(raw_data)
  num_batches = data.shape[0] // (batch_size * num_steps)
# makes sure the data ends with a full batch
  data = data [:num_batches*batch_size*num_steps:]
  data = np.reshape(data,(num_batches,batch_size,num_steps))
  print(data.shape)
  data = np.split(data, num_batches)
  data = [np.squeeze(subarray) for subarray in data]
  print(np.asarray(data).shape)
  # x = input, y = targets
  xdata = data
  ydata = np.copy(data)
  # shift all the targets by one: we want to predict the NEXT word
  for i in range(num_batches):
    ydata[i][:,:-1] = xdata[i][:,1:]
    ydata[i][:,-1] = 9999
  return np.asarray(xdata), ydata

# def ptb_producer(raw_data, batch_size, num_steps, name=None):
#   """Iterate on the raw PTB data.
#   This chunks up raw_data into batches of examples and returns Tensors that
#   are drawn from these batches.
#   Args:
#     raw_data: one of the raw data outputs from ptb_raw_data.
#     batch_size: int, the batch size.
#     num_steps: int, the number of unrolls.
#     name: the name of this operation (optional).
#   Returns:
#     A pair of Tensors, each shaped [batch_size, num_steps]. The second element
#     of the tuple is the same data time-shifted to the right by one.
#   Raises:
#     tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
#   """
#   with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
#     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

#     data_len = tf.size(raw_data)
#     batch_len = data_len // batch_size
#     data = tf.reshape(raw_data[0 : batch_size * batch_len],
#                       [batch_size, batch_len])

#     epoch_size = (batch_len - 1) // num_steps
#     assertion = tf.assert_positive(
#         epoch_size,
#         message="epoch_size == 0, decrease batch_size or num_steps")
#     with tf.control_dependencies([assertion]):
#       epoch_size = tf.identity(epoch_size, name="epoch_size")

#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#     x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
#     y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
#     return x, y