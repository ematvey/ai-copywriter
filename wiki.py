import os
import spacy
from spacy.tokens import Doc
import re

wikidata_dir = os.path.join('.', 'wikidata')
wikidoc_dir = os.path.join(wikidata_dir, 'wikidocs')
vocab_dir = os.path.join(wikidata_dir, 'vocab')
vocab_lexeme_fn = os.path.join(vocab_dir, 'vocab.bin')
vocab_strings_fn = os.path.join(vocab_dir, 'strings.json')
wikidoc_fn_template = 'wikidoc-%08d'

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
    en.vocab.load_lexemes(vocab_lexeme_fn)
    __nlp__ = en
  return __nlp__

def read_all(nlp_=None, epochs=1):
  if nlp_ is None:
    print('loading nlp object')
    nlp_ = nlp()
  files = [
      os.path.join(wikidoc_dir, fn)
      for fn in os.listdir(wikidoc_dir)
      if wikidoc_fn_re.match(fn) is not None
  ]
  print('reading from %s files' % len(files))
  for _ in range(epochs):
    for fn in files:
      docs = []
      with open(fn, 'rb') as f:
        for byte_str in Doc.read_bytes(f):
          docs.append(Doc(nlp_.vocab).from_bytes(byte_str))
      yield docs
