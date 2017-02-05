import argparse
import tqdm
import os
import spacy
from multiprocessing import cpu_count
from spacy.tokens import Doc
from textacy.corpora import WikiReader
from iotools import FilePoolWriter, BinaryFile
from wiki import *

parser = argparse.ArgumentParser()
parser.add_argument('wikidump_file')
args = parser.parse_args()

wikidump = args.wikidump_file

def preprocess(doc):
  for sent in doc.sents:
    pass

def save_vocab(vocab):
  print('saving vocab...')
  with open(vocab_strings_fn, 'w') as f:
    vocab.strings.dump(f)
  vocab.dump(vocab_lexeme_fn)
  print('vocab saved')

def parse_and_save():
  en = spacy.load('en')
  reader = WikiReader(wikidump)
  records = reader.records(limit=10)
  def section_texts_flat(records):
    for record in records:
      for section in record['sections']:
        yield section['text']
  pipe = en.pipe(section_texts_flat(records),
                 n_threads=2,
                 batch_size=100)
  # pipe = (en(txt) for txt in section_texts_flat(records))
  # with FilePoolWriter(wikidoc_dir, wikidoc_fn_template) as f:
  _fn = os.path.join(wikidoc_dir, wikidoc_fn_template % 0)
  with open(_fn, 'wb') as f:
    for i, doc in enumerate(tqdm.tqdm(pipe)):
      if len(doc._py_tokens) == 0:
        continue
      doc_bytes = doc.to_bytes()
      restored = Doc(en.vocab).from_bytes(doc_bytes)
      f.write(doc_bytes)
      # if i % 1000 == 0:
      #   save_vocab(en.vocab)
  # save_vocab(en.vocab)

  print('test 1')
  docs = []
  with open(_fn, 'rb') as f:
    for byte_str in Doc.read_bytes(f):
      docs.append(Doc(en.vocab).from_bytes(byte_str))

  print('test 2')
  docs = []
  en2 = spacy.load('en')
  with open(_fn, 'rb') as f:
    for byte_str in Doc.read_bytes(f):
      try:
        docs.append(Doc(en2.vocab).from_bytes(byte_str))
      except Exception as e:
        print(e)
        break

  # import IPython
  # IPython.embed()

if __name__ == '__main__':
  parse_and_save()
