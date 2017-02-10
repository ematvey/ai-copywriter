# AI copywriter
Given a large corpus of correct English texts and a one-way mechanism to corrupt a sentence in such a way that will be similar to mistakes that non-native speaker makes, we can develop seq2seq model that acts like denoising autoencoder. It could be used by non-proficient English writers to write better texts.

Main hypothesis is that corruption does not have to be very clever, just sufficiently diverse. Currently I use a set of simple heuristics for corruption, but later hope to develop a generative model for that.

## Status
Trying to make it work.
LSTMs seem to be less volatile then GRUs. Deeper nets (5 layers vs 3 layers) seem to converge to higher loss.

Probably unnecessary warning: this is research project, not in any way suitable for production use. No API stability guarantee of any kind is given. If you want to derive from this work, copy-paste it!

## How to run
Requires Python >=3.5 and TensorFlow r1.0.0.

Download Wikipedia dump e.g. with 

```wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2```

Install deps.

Run preprocessor:

```python3 wiki_prepare.py path/to/wikidump```

It should take several hours. Corruption types that preprocessor is doing could be found in wiki_data_op.py.

Run training:

```python3 worker.py --device=/gpu:0 --batch-size=30```

You may want to increase/decrease batch size based on your GPU mem. I have GTX 970 with 4gb.

## Related work
@atpaino [implemented similar model](https://github.com/atpaino/deep-text-corrector) on smaller corpus in Jan 2017. His work is based on tensorflow's `translate` example, which uses static RNN rollout with bucketizing. In this work I attempt to implement dynamic RNN rollout using tensorflow's lower-level blocks.

## License
MIT
