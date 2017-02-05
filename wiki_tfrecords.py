# gutenberg reader
import tqdm
import tensorflow as tf
output_fn = 'output_fn'

if False:
  import spacy
  from textacy.corpora import WikiReader

  wikidump = '/Users/ematvey/ai/copywriter/enwiki-20170201-pages-articles1.xml-p000000010p000030302.bz2'
  reader = WikiReader(wikidump)
  records = reader.records(limit=20)

  def section_texts_flat(records):
    for record in records:
      for section in record['sections']:
        yield section['text']

  en = spacy.load('en')
  pipe = en.pipe(section_texts_flat(records), entity=False, n_threads=4, batch_size=5)

  writer = tf.python_io.TFRecordWriter(output_fn)

  for doc in tqdm.tqdm(pipe):
    for sent in doc.sents:
      tokens = list(filter(lambda x: x.pos_ != 'SPACE', sent))
      tokens = [t.orth for t in tokens]
      ex = tf.train.Example(features=tf.train.Features(feature={
        'sent': tf.train.Feature(
          int64_list=tf.train.Int64List(value=tokens)
        ),
      }))
      writer.write(ex.SerializeToString())
  writer.close()

# test read
tf.reset_default_graph()
filenames = tf.train.string_input_producer([output_fn], num_epochs=1)
reader = tf.TFRecordReader()
_, example_ser = reader.read(filenames)
features = tf.parse_single_example(
    example_ser,
    # Defaults are not specified since both keys are required.
    features={
        'sent': tf.VarLenFeature(tf.int64),
    })
orig = features['sent']

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

sess = tf.InteractiveSession()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



import IPython
IPython.embed()