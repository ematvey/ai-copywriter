import argparse
import os
import pickle
import random
from multiprocessing import cpu_count

import numpy as np
import spacy
import tqdm
from spacy.tokens import Doc
from textacy.corpora import WikiReader
import nltk

train_dir = os.path.join('.', 'wikitrain')

vocab_size = 256 + 2


def _read_records(wikidump_file):
  reader = WikiReader(wikidump_file)
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
  return section_texts_flat(records)

DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve", "to"}

REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                "than": "then"}
from nltk.tokenize.moses import MosesDetokenizer
moses = MosesDetokenizer()

def corrupt_token_sequence(tokens):
  res = []
  for i, tok in enumerate(tokens):
    # sos = i == 0
    # eos = i == len(tokens)-1
    # if sos and random.random() > 0.9:
    #   res.append(tok.lower())
    #   continue
    # replacements
    replacement = REPLACEMENTS.get(tok)
    if replacement is not None and random.random() > 0.5:
      res.append(replacement)
      continue
    # drops
    if tok in DROPOUT_TOKENS and random.random() > 0.5:
      continue
    # just drop a word
    # if random.random() > 0.99:
    #   continue
    res.append(tok)
  return res

def reader(wikidump_file):
  records = _read_records(wikidump_file)
  for record in records:
    for sent in nltk.tokenize.sent_tokenize(record):
      tokens = nltk.tokenize.word_tokenize(sent)
      corrupted = []
      for _ in range(3):
        corrupted.append(moses.detokenize(corrupt_token_sequence(tokens), return_str=True))
      yield sent, corrupted
