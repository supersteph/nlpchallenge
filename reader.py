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

def _read_words_predict(filename):

  with open(filename, "r", encoding = "utf-8", errors = 'ignore') as f:
    a = f.read()
    a = a.replace("\n", " <eos> \n").replace("\t", " <eos> \n")
    a = a.split("\n")
    b = [sent.split() for sent in a[::2]]
    c = [sent.split() for sent in a[1::2]]
    return b,c

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
def _file_to_word_ids_predict(filename, word_to_id):
  eng, no = _read_words_predict(filename)
  b = eng[:]
  c = no
  for i in range(len(eng)):
    b[i] = [word_to_id[word] if word in word_to_id else 9999 for word in eng[i]]
    c[i] = [word_to_id[word] if word in word_to_id else 9999 for word in no[i]]

  return eng,b,c


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

  train_path = "gay.txt"

  word_to_id = _build_vocab(train_path)
  correct_data, wrong_data = _file_to_word_ids(train_path, word_to_id)
  # word_to_id = _build_vocab_single(train_path)
  # correct_data = _file_to_word_ids_single(train_path, word_to_id)
  vocabulary = len(word_to_id)
  return correct_data, wrong_data, vocabulary
  # return correct_data, word_to_id

def predict_raw_data(data_path=None):
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

  train_path = "train.txt"

  word_to_id = _build_vocab(train_path)
  words, correct_data, wrong_data = _file_to_word_ids_predict(train_path, word_to_id)
  # word_to_id = _build_vocab_single(train_path)
  # correct_data = _file_to_word_ids_single(train_path, word_to_id)
  return words, correct_data, wrong_data, word_to_id
  # return correct_data, word_to_id

def getinputoutput(raw_data, batch_size, num_steps):
  data = np.asarray(raw_data)
  num_batches = data.shape[0] // (batch_size)
# makes sure the data ends with a full batch
  data = data [:num_batches*batch_size:]
  data = np.reshape(data,(batch_size,num_batches))
  num_epochs = (num_batches-1)//num_steps
  # x = input, y = targets
  xdata = np.zeros(shape = (num_epochs,batch_size,num_steps))
  ydata = np.zeros(shape = (num_epochs,batch_size,num_steps))

  # shift all the targets by one: we want to predict the NEXT word
  for i in range(num_epochs):
    xdata[i] = data[:,i*num_steps:i*num_steps+num_steps]
    ydata[i] = data[:,i*num_steps+1:i*num_steps+num_steps+1]
  xdata = xdata[:num_batches-1]
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