#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
parser.add_argument('--wikidump', default='./wikidump')
parser.add_argument("--device", default="/cpu:0")
parser.add_argument('--checkpoint-frequency', type=int, default=100)
parser.add_argument('--eval-frequency', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=30)
parser.add_argument("--max-grad-norm", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.0001)
args = parser.parse_args()

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell
from model import Seq2SeqModel
import pandas as pd
import helpers
import wiki_char as wiki
import warnings
warnings.filterwarnings("ignore")

tf.reset_default_graph()
tf.set_random_seed(1)

train_dir = wiki.train_dir

checkpoint_dir = os.path.join(train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, 'model')
tflog_dir = os.path.join(train_dir, 'tflog')

if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)

def create_model(session, restore_only=False):
  # with bidirectional encoder, decoder state size should be
  # 2x encoder state size
  is_training = tf.placeholder(dtype=tf.bool, name='is_training')

  encoder_cell = LSTMCell(128)
  encoder_cell = MultiRNNCell([encoder_cell]*3)
  decoder_cell = LSTMCell(256)
  decoder_cell = MultiRNNCell([decoder_cell]*3)
  model = Seq2SeqModel(encoder_cell=encoder_cell,
                       decoder_cell=decoder_cell,
                       vocab_size=wiki.vocab_size,
                       embedding_size=30,
                       attention=True,
                       bidirectional=True,
                       is_training=is_training,
                       device=args.device,
                       debug=False)

  saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1)
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint:
    print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
    saver.restore(session, checkpoint.model_checkpoint_path)
  elif restore_only:
    raise FileNotFoundError("Cannot restore model")
  else:
    print("Created model with fresh parameters")
    session.run(tf.global_variables_initializer())
  tf.get_default_graph().finalize()
  return model, saver


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
  model, saver = create_model(session)
  summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())

  batch_size = args.batch_size

  def decode(seq):
    b = bytes([c-2 for c in seq if c > 1])
    try:
      s = b.decode('utf-8', errors='ignore')
      return s
    except UnicodeDecodeError:
      print('error decoding %s' % seq)
      return ''

  # queue = tf.RandomShuffleQueue(
  #   capacity=2**12,
  #   min_after_dequeue=batch_size,
  #   dtypes=[tf.uint8, tf.uint8],
  #   seed=12,
  #   shapes=[[None, None], [None], [None, None], [None]])

  # x_batch_enqueue = tf.placeholder(tf.uint8, shape=[None, None])
  # x_length_enqueue = tf.placeholder(tf.uint8, shape=[None])
  # y_batch_enqueue = tf.placeholder(tf.uint8, shape=[None, None])
  # y_length_enqueue = tf.placeholder(tf.uint8, shape=[None])

  # enqueue_op = queue.enqueue_many([x_batch_enqueue, x_length_enqueue, y_batch_enqueue, y_length_enqueue])

  def batch_iterator():
    data = wiki.reader(args.wikidump)
    def pull_batch():
      xs = []
      ys = []
      while len(xs) < batch_size:
        original, corrupted = next(data)
        if len(original) > 200:
          continue # skip long sentences
        original = [b+2 for b in bytes(original, encoding='utf-8')]
        corrupted = [[b+2 for b in bytes(corr, encoding='utf-8')] for corr in corrupted]
        xs.append(original)
        ys.append(original)
        for corr in corrupted:
          xs.append(corr)
          ys.append(original)
      return xs, ys
    while 1:
      xs, ys = pull_batch()
      yield xs, ys

  batches = batch_iterator()

  def queue_feeder():
    for input_seq, target_seq in batch_iterator():
      inputs_, inputs_length_ = helpers.batch(input_seq)
      targets_, targets_length_ = helpers.batch(target_seq)
      session.run(enqueue_op, {
        x_batch_enqueue: inputs_,
        x_length_enqueue: inputs_length_,
        y_batch_enqueue: targets_,
        y_length_enqueue: targets_length_,
      })

  def train_one():
    xs, ys = next(batches)
    fd = model.make_train_inputs(xs, ys)

    try:
      t0 = time.clock()
      step, _, loss, summaries = session.run([model.global_step, model.train_op, model.loss, model.summary_op], fd)
      td = time.clock() - t0

      summary_writer.add_summary(summaries, global_step=step)

      if step % 1 == 0:
        print('step %s, loss=%s, t=%s, inputs=%s, ' % (
          step, loss, round(td, 2), fd[model.encoder_inputs].shape,
        ))

      # if step % 5 == 0:
      #   for x, y in zip(xs, ys):
      #     print('> %s' % decode(x))
      #     print('< %s' % decode(y))

      # if step % 25 == 0:
      #   print('step %s, try decode' % step)
      #   out = session.run(model.decoder_prediction_inference, model.make_inference_inputs(xs))
      #   for x, y, o in zip(xs, ys, out.T):
      #     print('IN:      %s' % decode(x))
      #     print('OUT:     %s' % decode(o))
      #     print('TARGET:  %s' % decode(y))
      if step != 0 and step % args.checkpoint_frequency == 0:
        print('checkpoint & graph meta')
        saver.save(session, checkpoint_path, global_step=step)
        print('checkpoint done')
    except tf.errors.ResourceExhaustedError as e:
      print('REE: {}'.format(e))

  if args.mode == 'train':
    for i in range(1000000):
      try:
        train_one()
      except StopIteration:
        break
  elif args.mode == 'eval':
    # def decode(s):
    #   parsed = wiki.nlp()(s)
    #   tokens = np.array([t.orth for t in parsed], dtype=np.uint32)
    #   rev = wiki.normalize_sequence(tokens)
    #   fd = model.make_inference_inputs([rev])
    #   pred = session.run(model.decoder_prediction_inference, fd)
    #   words = []
    #   for p in pred[:, 0]:
    #     if p == 1:
    #       break
    #     elif p == 2:
    #       words.append('<UNK>')
    #     else:
    #       words.append(wiki.nlp().vocab[p-3].orth_)
    #   return ' '.join(words)
    import IPython
    IPython.embed()
