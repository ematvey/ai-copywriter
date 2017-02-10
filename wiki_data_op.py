import pickle
import random
import numpy as np


class Corruptor():
  _DETS = {"a", "an", "the"}
  _DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve", "to"}

  REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                  "than": "then"}

  def __init__(self, vocab, corrupted_n=10):
    self.corrupted_n = corrupted_n
    self.vocab = vocab

    self.dropout_tokens = {self.vocab[tok].orth for tok in self._DROPOUT_TOKENS}
    self.dets = [self.vocab[tok].orth for tok in self._DETS]

    self.comma = self.vocab[','].orth

  def __call__(self, tokens):
    res = []
    for i, tok in enumerate(tokens):
      sos = i == 0
      eos = i == len(tokens)-1

      if sos and random.random() > 0.9:
        res.append(tok.lower)
        continue

      # replacements
      replacement = self.REPLACEMENTS.get(tok.orth)
      if replacement is not None and random.random() > 0.7:
        res.append(replacement)
        continue

      # drops
      if tok.orth in self.dropout_tokens and random.random() > 0.7:
        continue

      # lemmatize
      if tok.lemma != tok.orth and random.random() > 0.95:
        res.append(tok.lemma)
        continue

      next_tok = None
      if not eos:
        next_tok = tokens[i+1]

      if (next_tok is not None and
          tok.pos_ != 'DET' and
          next_tok.pos_ in ('NOUN', 'PRON') and
          random.random() > 0.7):
        res.append(random.choice(self.dets))
        continue

      if not eos and tok.pos_ == 'PUNCT' and random.random() > 0.7:
        continue

      # add random comma (and proceed)
      if tok.pos_ != 'PUNCT' and random.random() > 0.97:
        res.append(self.comma)

      # just drop a word
      if random.random() > 0.97:
        continue

      res.append(tok.orth)

    return res

class Preprocessor():
  def __init__(self, vocab, corrupted_n=10):
    self.corruptor = Corruptor(vocab, corrupted_n)

  @staticmethod
  def pack_sent(sent):
    return np.array(sent, np.uint32).tobytes()

  @staticmethod
  def unpack_sent(byte_string):
    return np.fromstring(byte_string, dtype=np.uint32)

  def _preprocess(self, sent):
    sent = [t for t in sent if t.pos_ != 'SPACE']
    corrupted = [self.corruptor(sent) for _ in range(self.corrupted_n)]
    tokens = [t.orth for t in sent]
    return tokens, corrupted

  def pack(self, sent):
    tokens, corrupted = self._preprocess(sent)
    return pickle.dumps((
        self.pack_sent(tokens),
        [self.pack_sent(ct) for ct in corrupted]
    ))

  def unpack(self, byte_string):
    tokens, corrupted = pickle.loads(byte_string)
    tokens = self.unpack_sent(tokens)
    corrupted = [self.unpack_sent(ts) for ts in corrupted]
    return tokens, corrupted

  def unpack_repr(self, byte_string):
    tokens, corrupted = self.unpack(byte_string)
    repr = lambda x: ' '.join([self.vocab[t].orth_ for t in x])
    print('original:  ', repr(tokens))
    for s in corrupted:
      print('corrupted: ', repr(s))
