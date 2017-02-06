# gutenberg reader
from collections import defaultdict
import glob
import tqdm
import spacy
gutenberg_path = '/Users/ematvey/www.gutenberg.lib.md.us/1'

import zipfile
import textacy

def reader():
  paths = list(textacy.fileio.utils.get_filenames(gutenberg_path, recursive=True))
  for path in tqdm.tqdm(paths, total=len(paths)):
    with open(path, 'rb') as f:
      with zipfile.ZipFile(f) as zf:
        for name in zf.namelist():
          data = zf.read(name)
          data = data.decode('utf-8', errors='replace')
          data = textacy.preprocess_text(data, fix_unicode=True, no_numbers=True, transliterate=True)
          yield data

en = spacy.load('en')

# def build_vocab():
#   vocab = defaultdict(int)
#   for book in read_books():
#     tokens = en.tokenizer(book)
#     for token in tokens:
#       vocab[token.orth_] += 1
#   return vocab

def reader():
  for txt in [open('nietzsche.txt', 'r').read()]:
    yield txt

try:
  # vocab = build_vocab()
  vocab = defaultdict(int)
  for book in reader():
    tokens = en.tokenizer(book)
    for token in tokens:
      vocab[token.orth_] += 1

finally:
  import pandas as pd

  s = pd.Series(vocab)
  s.sort_values()
  import IPython; IPython.embed()


python
text = """Vanity is one of the things which are perhaps most difficult for a noble man to understand: he will be tempted to deny it, where another kind of man thinks he sees it self-evidently. The problem for him is to represent to his mind beings who seekto arouse a good opinion of themselves which they themselves do not possess--and consequently also do not "deserve,"--and who yet BELIEVE in this good opinionafterwards."""
# verbatim portion of http://www.gutenberg.org/cache/epub/4363/pg4363.txt

import spacy
en = spacy.load('en')
doc = en(doc)
for token in doc:
  if token.pos_ == 'X':
    print(token.orth_)