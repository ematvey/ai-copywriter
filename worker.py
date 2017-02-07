import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-frequency', type=int, default=100)
parser.add_argument('--eval-frequency', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=30)
parser.add_argument("--device", default="/cpu:0")
parser.add_argument("--max-grad-norm", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell
from model import Seq2SeqModel, train_on_copy_task
import pandas as pd
import helpers
import wiki
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

  encoder_cell = LSTMCell(64)
  encoder_cell = MultiRNNCell([encoder_cell]*5)
  decoder_cell = LSTMCell(128)
  decoder_cell = MultiRNNCell([decoder_cell]*5)
  model = Seq2SeqModel(encoder_cell=encoder_cell,
                       decoder_cell=decoder_cell,
                       vocab_size=wiki.vocab_size,
                       embedding_size=300,
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
  data = wiki.read_trainset(epochs=5)

  def pull_batch():
    xs = []
    ys = []
    while len(xs) < batch_size:
      i, j, l, fn, (original, corrupted) = next(data)
      if len(original) > 30:
        continue # skip long sentences
      original = wiki.normalize_sequence(original)
      corrupted = [wiki.normalize_sequence(s) for s in corrupted]

      xs.append(original)
      ys.append(original)
      for c in corrupted:
        xs.append(c)
        ys.append(original)
    return xs, ys, i, j, l, fn

  def train_one():
    xs, ys, i, j, l, fn = pull_batch()
    fd = model.make_train_inputs(xs, ys)

    try:
      t0 = time.clock()
      step, _, loss, summaries = session.run([model.global_step, model.train_op, model.loss, model.summary_op], fd)
      td = time.clock() - t0

      summary_writer.add_summary(summaries, global_step=step)

      if step % 1 == 0:
        print('step %s, loss=%s, t=%s, inputs=%s, file=%s of %s, epoch=%s, fn=%s' % (
          step, loss, round(td, 2), fd[model.encoder_inputs].shape, j, l, i, fn
        ))
      if step != 0 and step % args.checkpoint_frequency == 0:
        print('checkpoint & graph meta')
        saver.save(session, checkpoint_path, global_step=step)
        print('checkpoint done')
    except tf.errors.ResourceExhaustedError as e:
      print('REE: {}'.format(e))


  for i in range(1000000):
    try:
      train_one()
    except StopIteration:
      break
