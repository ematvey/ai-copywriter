import os
import random

class BinarySequenceFile():
  """Allows packing multible blobs into single file,
     takes care of addressing."""
  def __init__(self, name, mode='rb'):
    self.name = name
    self.mode = mode
    self.length_bytesize = 4
    self.max_length = 2 ** (self.length_bytesize * 8)
    self._file = open(self.name, self.mode)

  def write(self, byte_string):
    assert isinstance(byte_string, bytes)
    size = len(byte_string)
    assert size < self.max_length
    size_encoded = size.to_bytes(self.length_bytesize, 'big')
    self._file.write(size_encoded)
    written_size = self._file.write(byte_string)
    assert written_size == size

  def read(self):
    """Read one"""
    size_encoded = self._file.read(self.length_bytesize)
    size = int.from_bytes(size_encoded, 'big')
    data = self._file.read(size)
    if data == b'':
      raise EOFError()
    return data

  def __iter__(self):
    while True:
      try:
        yield self.read()
      except EOFError:
        raise StopIteration()

  def __enter__(self):
    return self

  def close(self):
    self._file.close()

  def __exit__(self, ex_type, ex_value, ex_trace):
    self.close()
    if ex_value is not None:
      raise ex_value

class FilePoolWriter():
  def __init__(self, dir_path, fn_template, poolsize=10, filesize=10000):
    self.poolsize = poolsize
    self.filesize = filesize
    self.dir_path = dir_path

    self.fn_template = fn_template
    self.i = 0
    self.pool = []

  def _open_new_file(self):
    fn = os.path.join(os.path.expandvars(self.dir_path), self.fn_template % self.i)
    f = {'file': BinarySequenceFile(fn, 'wb'), 'count': 0}
    print('opened file', fn)
    self.i += 1
    return f

  def write(self, byte_string):
    f_idx = random.randint(0, self.poolsize-1)
    f = self.pool[f_idx]
    f['file'].write(byte_string)
    f['count'] += 1
    if f['count'] >= self.filesize:
      print('file %s contains %s records, rotating' % (f['file'].name, f['count']))
      f['file'].close()
      self.pool[f_idx] = self._open_new_file()

  def __enter__(self):
    while len(self.pool) < self.poolsize:
      self.pool.append(self._open_new_file())
    assert len(self.pool) == self.poolsize
    return self

  def close(self):
    for f in self.pool:
      print('closing file %s after %s writes' % (f['file'].name, f['count']))
      f['file'].close()
    self.pool = []

  def __exit__(self, ex_type, ex_value, ex_trace):
    self.close()
    if ex_value is not None:
      raise ex_value
