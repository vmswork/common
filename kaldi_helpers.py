import numpy as np
import sys
import os
import inspect

class error(Exception):
  pass

class ark_error(error):
  """Exception raised for errors in the input contents.
  """
  pass

class kaldi_data:
  def __init__(self, input, goal = 'r'):
    """ Helper class to read and write data in ark.t format.
        Don't use it other than inside WITH AS clause/
    
    """
    self._file = input
    self._out_utterance_name = ''
    try:
      if self._file == "ark,t:-":
        if goal in ['w', 'a']:
          self._input_stream = sys.stdout
        else:
          self._input_stream = sys.stdin
      else:
        self._input_stream = open(self._file, goal)
    except IOError:
      print('Error: cannot open %s' % (self._file))
  def __clean(self):
    if hasattr(self, '_input_stream'):
      # If we tried to write something in this session with batches
      # there is no ']' after the last row of last batch.
      # So we're gonna add it
      if self._out_utterance_name:
        # Finalize last utterance
        self.__finalize_utterance()      
      self._input_stream.close()
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc_value, traceback):
    self.__clean()
  def read_utterance(self, batch_size=-1):
    """ Read input utterance by utterance.
        batch_size: number of frames to be returned in a single call. 
                    Last batch may be smaller than batch_size
                    -1 means return whole utterance in one gulp
    """
    i = self._input_stream
    while True:
      line = i.readline()
      if not line:
        break
      utterance_name = self.__decode_utterance_name(line)
      contents = self.__read_body(batch_size)
      for b in contents:
        yield [utterance_name, b]
  def read_counts(self):
    """ Read kaldi occs file and return numpy array with logarithms.
            File is assumed to contain a single row starting with '[', 
            ending with ']' and with counts in between separated by spaces.
    """
    i = self._input_stream
    line = i.readline()
    [a, _] = self.__decode_body(line)
    a = self.__counts2logs(a)
    return a
  def write_utterance(self, data):
    """ Writes in ark,t format..
        data: a list of lists. Each element is expected to contatin 2 elements: 
              an utterance name and a numpy array with the data.
              Example: [['name1', data1], ['name2', data2]]
    """    
    o = self._input_stream
    for a in data:
      self.write_utterance_name(a[0])
      self.__write_body(a[1])
  def write_utterance_name(self, utterance_name):
    o = self._input_stream
    o.write(self.__encode_utterance_name(utterance_name))
  def write_batches(self, data):
    o = self._input_stream
    if data[0] != self._out_utterance_name:
      # If current utterance is not the first one
      if self._out_utterance_name:
        # Finalize previous utterance
        self.__finalize_utterance()
      # In any case output utterance name with '['
      self.write_utterance_name(data[0])
      self._out_utterance_name = data[0]
    self.__write_body(data[1], finalize_utterance=False)
      
  def __encode_utterance_name(self, name):
    return name + ' ['
  def __finalize_utterance(self):
    o = self._input_stream
    s = ' ]\n'
    o.write(s)
  def __write_body(self, data, finalize_utterance=True):
    o = self._input_stream
    for a in data:
      s = '\n' + ' '.join(map(str, a))
      o.write(s)
    if finalize_utterance:
      self.__finalize_utterance()
  def __counts2logs(self, a):
    return np.log(a) - np.log(a.sum())
  def __decode_utterance_name(self, line):
    """ Decodes utterance name.
        line: first string of an utterance. It is expected to look like
              '011_011C0201_PED_SIMU  ['
    """
    a = line.strip().split(' ')
    if (not a[0]) or (a[-1] != '['):
      raise ark_error(line)
    return a[0]
  def __read_body(self, batch_size):
    """ Reads and decodes contents switched between '<utt_name> [' and ']'
        batch_size: number of frames to be returned in a single call. 
                    Last batch may be smaller than batch_size
                    -1 means return whole utterance in one batch
    """
    i = self._input_stream
    contents = []
    is_last = False
    while not is_last:
      line = i.readline()
      if not line:
        raise ark_error('Unexpected end of input')
      [a, is_last] = self.__decode_body(line)
      self.__accumulate_body(contents, a)
      do_flush = self.__check_flush(contents, batch_size, is_last)
      if do_flush:
        yield np.array(contents)
        contents = []
  def __decode_body(self, line):
    """ Decodes ark contentes
        line: a text string containing floats delimited with spaces
              and with possible ']' as last character.
    """
    is_last = False
    a = line.strip().split(' ')
    if a[0] == '[':
      a = a[1:]
    if a[-1] == ']':
      is_last = True
      a = a[:-1]
    return np.array(a,dtype='float32'), is_last
  def __accumulate_body(self, contents, a):
    contents.append(a)
    # return np.array([contents, a])
  def __check_flush(self, contents, batch_size, is_last):
    if (len(contents) == batch_size) or is_last:
      return True
    return False
  
  @property
  def input(self):
    return self._file
  @input.setter
  def input(self, value):
    self._file = value
    
def tprint(message):
  curframe = inspect.currentframe()
  calframe = inspect.getouterframes(curframe, 2)
  print(calframe[1][3] + ': ' + message)

def test_init():
  tprint('...')
  existing_file = './tests/kaldi_helpers/2.ark'
  for _ in range(2):
    with kaldi_data(existing_file) as kd:
      tprint("Existing file %s is opened" % (existing_file))
  tprint('Opening non-existing file...')
  nonexisting_file = './this_file_does_not_exist'
  with kaldi_data(nonexisting_file) as kd:
      tprint("Non-existing file %s is opened" % (nonexisting_file))
def test_read_utterance():
  tprint('...')
  batch_size = -1
  existing_file = './tests/kaldi_helpers/2.ark'
  names = ['011_011C0201_PED_SIMU', '011_011C0202_PED_SIMU']
  sizes = [653, 694]
  with kaldi_data(existing_file) as kd:
    batch1 = kd.read_utterance(batch_size)
    for i in range(2):
      data = batch1.next()
      tprint(str(data[0]))
      assert data[0] == names[i], '%s != %s' % (data[0], names[i])
      tprint(str(np.shape(data[1])))
      assert np.shape(data[1])[0] == sizes[i], '%d != %d' % (np.shape(data[1])[0], sizes[i])
  tprint('OK')
def test_read_utterance_stdin():
  tprint('...')
  batch_size = -1
  file = 'ark,t:-'
  names = ['011_011C0201_PED_SIMU', '011_011C0202_PED_SIMU']
  sizes = [653, 694]
  with kaldi_data(file) as kd:
    batch1 = kd.read_utterance(batch_size)
    for i in range(2):
      data = batch1.next()
      tprint(str(data[0]))
      assert data[0] == names[i], '%s != %s' % (data[0], names[i])
      tprint(str(np.shape(data[1])))
      assert np.shape(data[1])[0] == sizes[i], '%d != %d' \
                                % (np.shape(data[1])[0], sizes[i])  
  tprint('OK')
def test_read_counts():
  tprint('...')
  file = './tests/kaldi_helpers/final.occs'
  with kaldi_data(file) as kd:
    counts = kd.read_counts()
    assert len(counts) == 1956, 'len(counts) != 1956'
    assert counts[0] - (-2.84299) < .00001, 'counts[0] == -2.84299'
  tprint('OK')
def test_write_utterance():
  tprint('...')
  file = './tests/kaldi_helpers/2out.ark'
  a = np.random.rand(2, 10, 100)
  data = []
  data.append(['sent1', a[0]])
  data.append(['sent2', a[1]])
  with kaldi_data(file, 'w') as kd:
    kd.write_utterance(data)
  with kaldi_data(file, 'r') as kd:
    sent = kd.read_utterance(-1)
    for i in range(len(data)):
      input_data = sent.next()
      assert input_data[0] == data[i][0], 'input_data[0] != data[i][0]'
      gt_data_sqsum = (data[i][1]*data[i][1]).sum()
      input_data_sqsum = (input_data[1]*input_data[1]).sum()
      assert  (gt_data_sqsum - input_data_sqsum) < 0.0001, \
                      'gt_data_sqsum != input_data_sqsum (%.5f, %.5f)'\
                      % (gt_data_sqsum, input_data_sqsum)
  tprint('OK')
def test_append_utterance():
  tprint('...')
  file = './tests/kaldi_helpers/2out.ark'
  a = np.random.rand(2, 10, 100)
  data = []
  data.append(['sent1', a[0]])
  data.append(['sent2', a[1]])
  with kaldi_data(file, 'w') as kd:
    kd.write_utterance(data)
  with kaldi_data(file, 'a') as kd:
    kd.write_utterance(data)
  with kaldi_data(file, 'r') as kd:
    r = [d for d in kd.read_utterance(-1)]
  assert len(r) == len(data)*2, 'len(r) != len(data)*2 (%d, %d)' % (len(r), len(data)*2)
  tprint('OK')  
def test_write_batches():
  tprint('...')
  file = './tests/kaldi_helpers/2out_batches.ark'
  a = np.random.rand(2, 10, 100)
  data = []
  data.append(['sent1', a[0]])
  data.append(['sent2', a[1]])
  batch_size = 2
  with kaldi_data(file, 'w') as kd:
    for d in data:
      for i in range(0, len(d[1]), batch_size):
        kd.write_batches([d[0], d[1][i:i+batch_size]])
  with kaldi_data(file, 'r') as kd:
    sent = kd.read_utterance(-1)
    for i in range(len(data)):
      input_data = sent.next()
      assert input_data[0] == data[i][0], 'input_data[0] != data[i][0]'
      gt_data_sqsum = (data[i][1]*data[i][1]).sum()
      input_data_sqsum = (input_data[1]*input_data[1]).sum()
      assert  (gt_data_sqsum - input_data_sqsum) < 0.0001, \
                      'gt_data_sqsum != input_data_sqsum (%.5f, %.5f)'\
                      % (gt_data_sqsum, input_data_sqsum)
  
  
  tprint('OK')
      
def run_tests():
  tprint('Running tests...')
  test_init()
  test_read_utterance()
  # test_read_utterance_stdin()
  test_read_counts()
  test_write_utterance()
  test_append_utterance()
  test_write_batches()   

  
if __name__ == '__main__':
  run_tests()
  