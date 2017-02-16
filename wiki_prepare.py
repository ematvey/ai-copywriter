import argparse
import os
import pickle
import random
from multiprocessing import cpu_count

import numpy as np
import spacy
import tqdm
from spacy.tokens import Doc
from wiki_reader import WikiReader

from iotools import BinarySequenceFile, FilePoolWriter
from wiki import *
from wiki_data_op import Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument('wikidump_file')
args = parser.parse_args()

wikidump = args.wikidump_file

def save_vocab(vocab):
  print('saving vocab...')
  with open(vocab_strings_fn, 'w') as f:
    vocab.strings.dump(f)
  vocab.dump(vocab_lexeme_fn)
  print('vocab saved')

def parse_and_save():
  en = spacy.load('en')
  reader = WikiReader(wikidump)
  records = reader.records()
  def section_texts_flat(records):
    while 1:
      try:
        record = next(records)
      except OSError as e:
        print('error: %s' % e)
      else:
        for section in record['sections']:
          yield section['text']
  pipe = en.pipe(section_texts_flat(records),
                 n_threads=cpu_count(),
                 batch_size=1000)
  # pipe = (en(txt) for txt in section_texts_flat(records))
  preproc = Preprocessor(en.vocab)
  with FilePoolWriter(wikidoc_dir, wikidoc_fn_template) as f:
    for i, doc in enumerate(tqdm.tqdm(pipe)):
      if len(doc._py_tokens) <= 7:
        # short sentences -- nah
        continue
      for sent in doc.sents:
        packed = preproc.pack(sent)
        f.write(packed)
      if i % 10000 == 0:
        print('i=%s, saving vocab' % i)
        save_vocab(en.vocab)
  save_vocab(en.vocab)
  import IPython
  IPython.embed()

if __name__ == '__main__':
  parse_and_save()
