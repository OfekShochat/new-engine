import torch
import numpy as np
from torch.utils.data import IterableDataset
from os import path
from glob import glob

BATCH = 1_000_000

class DataManager(IterableDataset):
  def __init__(self, data_dir) -> None:
    self.data_dir = data_dir
    
    self.data_ = np.array([])
    self.targets_ = np.array([])
    self.offset = 0
    self.seen_files = {}
  
  
  def read_npy_chunk(self, filename, start_row, num_rows):
    """
    gistfile1.py from dwf (https://gist.github.com/dwf/1766222 - https://github.com/dwf)
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
      major, minor = np.lib.format.read_magic(fhandle)
      shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
      assert not fortran, "Fortran order arrays not supported"
      # Make sure the offsets aren't invalid.
      assert start_row < shape[0], (
        'start_row is beyond end of file'
      )
      assert start_row + num_rows <= shape[0], (
        'start_row + num_rows > shape[0]'
      )
      # Get the number of elements in one 'row' by taking
      # a product over all other dimensions.
      row_size = np.prod(shape[1:])
      start_byte = start_row * row_size * dtype.itemsize
      fhandle.seek(int(start_byte), 1)
      n_items = row_size * num_rows
      flat = np.fromfile(fhandle, count=int(n_items), dtype=dtype)
      return flat #.reshape((-1,) + shape[1:])

  def readBatch(self):
    files = glob(path.join(self.data_dir, "*.npy"))
    for f in files:
      if 'data' in f.split(path.sep)[-1]:
        try:
          self.data_ = self.read_npy_chunk(f, self.getAndIncrememntSeen(f) * BATCH, BATCH)
          self.targets_ = self.read_npy_chunk(path.join(self.data_dir, self.getTargetsFilename(f)), self.getAndIncrememntSeen(f) * BATCH, BATCH)
        except AssertionError as e:
          continue
        except FileNotFoundError as e:
          continue
    print("loaded {} fens and {} results".format(len(self.data_), len(self.targets_)))

  def getAndIncrememntSeen(self, f):
    try:
      self.seen_files[f] += 1
      return self.seen_files[f] - 1
    except KeyError:
      self.seen_files[f] = 1
      return self.seen_files[f] - 1

  def getTargetsFilename(self, f):
    return f.split(path.sep)[-1].replace("data", "targets")

  def resetSeenFiles(self):
    self.seen_files = {}

  def toPlanes(self, x):
    PawnKing = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]
    QueenRook = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]
    KnightBishop = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]

    i = 0
    if "w" in x:
      color = 1
    else:
      color = -1

    for piece in reversed(x[:x.find(" ")]):
      if piece == "P":
        PawnKing[0][i] = 1
      elif piece == "p":
        PawnKing[1][i] = 1
      elif piece ==  "K":
        PawnKing[2][i] = 1
      elif piece ==  "k":
        PawnKing[3][i] = 1
      
      elif piece ==  "Q":
        QueenRook[0][i] = 1
      elif piece ==  "q":
        QueenRook[1][i] = 1
      elif piece ==  "R":
        QueenRook[2][i] = 1
      elif piece ==  "r":
        QueenRook[3][i] = 1

      elif piece ==  "N":
        KnightBishop[0][i] = 1
      elif piece ==  "n":
        KnightBishop[1][i] = 1
      elif piece ==  "B":
        KnightBishop[2][i] = 1
      elif piece ==  "b":
        KnightBishop[3][i] = 1
      elif piece == "8":
        i += 7
      elif piece == "/":
        continue
      elif piece == " ":
        break
      i += 1
    assert color != 0
    return (torch.FloatTensor(PawnKing), torch.FloatTensor(QueenRook), torch.FloatTensor(KnightBishop)), color

  def __len__(self):
    assert len(self.data_) == len(self.targets_)
    return len(self.targets_)
  
  def sample_iter(self):
    info = torch.utils.data.get_worker_info()
    worker_id = info.id
    per_worker = int(len(self)/info.num_workers)
    for idx in range(worker_id * per_worker, (worker_id + 1) * per_worker):
      d = self.toPlanes(self.data_[idx])
      yield d[0], self.targets_[idx] if d[1] == 1 else 1 - self.targets_[idx]

  def __iter__(self):
    for sample in self.sample_iter():
      yield sample[0], sample[1]