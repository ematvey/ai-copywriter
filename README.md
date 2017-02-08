# AI copywriter
Given a large corpus of correct English texts and a one-way mechanism to corrupt a sentence in such a way that will be similar to mistakes that non-native speaker makes, we can develop seq2seq model that acts like denoising autoencoder. It could be used by non-proficient English writers to write better texts.

Main hypothesis is that corruption does not have to be very clever, just sufficiently diverse. Currently I use a set of simple heuristics for corruption, but later hope to develop a generative model for that.

## Status
Trying to make it work.
LSTMs seem to be less volatile then GRUs. Deeper nets (5 layers vs 3 layers) seem to converge to higher loss.

## How to run
Requires Python >=3.5 and TensorFlow r1.0.0.

Download Wikipedia dump from `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2`.

Install deps.

Run preprocessor: `python3 wiki_prepare.py path/to/wikidump`.

Run training: `python3 worker.py --device=/gpu:0 --batch-size=30`.

Batch size may be increased/decreased based on your GPU mem.
