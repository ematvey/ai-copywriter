import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from model_new import Seq2SeqModel, train_on_copy_task
import pandas as pd
import helpers
import wiki
import warnings
warnings.filterwarnings("ignore")

tf.reset_default_graph()
tf.set_random_seed(1)

data = wiki.read_trainset()
print(next(data))
print(next(data))
print(next(data))

with tf.Session() as session:
  # with bidirectional encoder, decoder state size should be
  # 2x encoder state size
  model = Seq2SeqModel(encoder_cell=GRUCell(64),
                       decoder_cell=GRUCell(128),
                       vocab_size=wiki.vocab_size,
                       embedding_size=300,
                       attention=True,
                       bidirectional=True,
                       debug=False)

  session.run(tf.global_variables_initializer())

  batch_size = 100
  data = wiki.read_trainset()

  def pull_batch():
    xs = []
    ys = []
    while len(xs) < batch_size:
      original, corrupted = next(data)
      original = wiki.normalize_sequence(original)
      corrupted = [wiki.normalize_sequence(s) for s in corrupted]

      xs.append(original)
      ys.append(original)
      for c in corrupted:
        xs.append(c)
        ys.append(original)
    return xs, ys

  def train_one():
    xs, ys = pull_batch()
    fd = model.make_train_inputs(xs, ys)
    _, l = session.run([model.train_op, model.loss], fd)
    print(l)

  while True:
    try:
      train_one()
    except StopIteration:
      break


  # train_on_copy_task(session, model,
  #                     length_from=3, length_to=8,
  #                     vocab_lower=2, vocab_upper=10,
  #                     batch_size=100,
  #                     max_batches=3000,
  #                     batches_in_epoch=1000,
  #                     verbose=True)