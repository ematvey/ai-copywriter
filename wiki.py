import os
import spacy
import re
import iotools
import numpy as np
import random
from wiki_data_op import Preprocessor

wikidata_dir = os.path.join('.', 'wikidata')
wikidoc_dir = os.path.join(wikidata_dir, 'wikidocs')
vocab_dir = os.path.join(wikidata_dir, 'vocab')
vocab_lexeme_fn = os.path.join(vocab_dir, 'vocab.bin')
vocab_strings_fn = os.path.join(vocab_dir, 'strings.json')
wikidoc_fn_template = 'wikidoc-%08d'

train_dir = os.path.join('.', 'wikitrain')

wikidoc_fn_re = re.compile('^wikidoc-[0-9]{8}$')

for path in [wikidata_dir, wikidoc_dir, vocab_dir]:
  if not os.path.exists(path):
      os.makedirs(path)

__nlp__ = None
def nlp():
  global __nlp__
  if __nlp__ is None:
    en = spacy.load('en')
    with open(vocab_strings_fn, 'r') as f:
      en.vocab.strings.load(f)
    # en.vocab.load_lexemes(vocab_lexeme_fn)
    __nlp__ = en
  return __nlp__

files = [
    os.path.join(wikidoc_dir, fn)
    for fn in os.listdir(wikidoc_dir)
    if wikidoc_fn_re.match(fn) is not None
]
train_files = files[:int(len(files)*0.85)]
random.shuffle(train_files)

dev_files = files[int(len(files)*0.85):int(len(files)*0.95)]
test_files = files[int(len(files)*0.85):int(len(files)*0.95)]

vocab_size = 100000

def normalize_sequence(seq: np.ndarray, oov_token=2, reserved_tokens=3, max_vocab=vocab_size):
  seq = seq + 3
  seq[seq > max_vocab-1] = oov_token
  return seq

def _read(files, nlp_=None, epochs=1):
  if nlp_ is None:
    print('loading nlp object')
    nlp_ = nlp()
  l = len(files)
  print('reading from %s files' % l)
  preproc = Preprocessor(nlp_.vocab)
  for i in range(epochs):
    for j, fn in enumerate(files):
      with iotools.BinarySequenceFile(fn, 'rb') as f:
        for binstr in f:
          yield i, j, l, fn, preproc.unpack(binstr)

def read_trainset(epochs=1):
  return _read(train_files, epochs=epochs)

def read_devset(epochs=1):
  return _read(dev_files, epochs=epochs)