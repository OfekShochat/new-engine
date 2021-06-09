import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
from time import time
import numpy as np
import torch.optim as optim
import os
from sys import argv
from torch.optim.lr_scheduler import ExponentialLR
import psutil
import random

L1 = 512
L2 = 32

if not os.path.isdir("networks"):
  os.mkdir("networks")

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    """
    ARCH:                      _
      PawnKingMatrix     -> fc  \
      QueenRookMatrix    -> fc -> add -> fc -> fc
      KnightBishopMatrix -> fc _/
    """
    
    # subnetworks
    self.SubPawnKing_dense     = nn.Linear(4 * 64, 64)
    self.SubQueenRook_dense    = nn.Linear(4 * 64, 64)
    self.SubKnightBishop_dense = nn.Linear(4 * 64, 64)

    #mainNet
    self.MainInput_dense  = nn.Linear(64, L2)
    self.MainOut_dense    = nn.Linear(L2, 1)

    #self.dropout1 = nn.Dropout(p = 0.5)
    #self.dropout2 = nn.Dropout(p = 0.8)

    def init_weights(m):
      a = np.random.random()*2-1
      return a

    self.apply(init_weights)

    WPAWN = 0
    WKNIGHT = 1
    WBISHOP = 2
    WROOK = 3
    WQUEEN = 4
    WKING = 5
    BPAWN = 6
    BKNIGHT = 7
    BBISHOP = 8
    BROOK = 9
    BQUEEN = 10
    BKING = 11

    self.d = {
      "P":WPAWN,
      "N":WKNIGHT,
      "B":WBISHOP,
      "R":WROOK,
      "Q":WQUEEN,
      "K":WKING,

      "p":BPAWN,
      "n":BKNIGHT,
      "b":BBISHOP,
      "r":BROOK,
      "q":BQUEEN,
      "k":BKING
    }

  def forward(self, PawnKing, QueenRook, KnightBishop):
    PawnKing     = self.SubPawnKing_dense(PawnKing)
    QueenRook    = self.SubQueenRook_dense(QueenRook)
    KnightBishop = self.SubKnightBishop_dense(KnightBishop)
    added = torch.add(PawnKing, QueenRook)
    added = torch.add(added, KnightBishop)

    x  = self.MainInput_dense(added)
    x  = self.MainOut_dense(x)
    return torch.sigmoid(x)
  
  def toPlanes(self, x):
    PawnKing = np.zeros((4, 64, 1), dtype=np.float)
    QueenRook = np.zeros((4, 64, 1), dtype=np.float)
    KnightBishop = np.zeros((4, 64, 1), dtype=np.float)
    for i in range(8):
      for j in range(8):
        piece = str(x.piece_at(chess.SQUARES[i*8+j]))
        if piece != "None":
          if piece == "P":
            PawnKing[0][i*8+j] = 1
          elif piece == "p":
            PawnKing[1][i*8+j] = 1
          elif piece ==  "K":
            PawnKing[2][i*8+j] = 1
          elif piece ==  "k":
            PawnKing[3][i*8+j] = 1
          
          elif piece ==  "Q":
            QueenRook[0][i*8+j] = 1
          elif piece ==  "q":
            QueenRook[1][i*8+j] = 1
          elif piece ==  "R":
            QueenRook[2][i*8+j] = 1
          elif piece ==  "r":
            QueenRook[3][i*8+j] = 1

          elif piece ==  "N":
            KnightBishop[0][i*8+j] = 1
          elif piece ==  "n":
            KnightBishop[1][i*8+j] = 1
          elif piece ==  "B":
            KnightBishop[2][i*8+j] = 1
          elif piece ==  "b":
            KnightBishop[3][i*8+j] = 1
          else:
            print("we shouldnt get here")
    return PawnKing, QueenRook, KnightBishop

class DataManager(Dataset):
  def __init__(self, root) -> None:
    self.data_ = np.array([])
    self.targets_ = np.array([], dtype=np.int)
    self.offset = 0
    self.seen_files = {}
    #self.trainingData = np.memmap(filename='.trainingData.mapped', mode='w+', shape=(14, 12, 64))

    WPAWN = 0
    WKNIGHT = 1
    WBISHOP = 2
    WROOK = 3
    WQUEEN = 4
    WKING = 5
    BPAWN = 6
    BKNIGHT = 7
    BBISHOP = 8
    BROOK = 9
    BQUEEN = 10
    BKING = 11

    self.d = {
      "P":WPAWN,
      "N":WKNIGHT,
      "B":WBISHOP,
      "R":WROOK,
      "Q":WQUEEN,
      "K":WKING,

      "p":BPAWN,
      "n":BKNIGHT,
      "b":BBISHOP,
      "r":BROOK,
      "q":BQUEEN,
      "k":BKING
    }
  
  def read(self, f, batchIdx):
    if batchIdx * 2_000_000 > len(np.load(f, "r+", False)):
      raise Exception("dataBatchBuffer is bigger than the data file")
    f_len = len(np.load(f, "r+", False)[batchIdx * 2_000_000: (batchIdx+1) * 2_000_000])
    if f_len < 200_000:
      raise Exception("too little data left in file {} ({})".format(f, f_len))
    if 'data' in f:
      self.data_ = np.concatenate((np.load(f, "r+", False)[batchIdx * 2_000_000: (batchIdx+1) * 2_000_000], self.data_))
    else:
      self.targets_ = np.concatenate((np.load(f, "r+", False)[batchIdx * 2_000_000: (batchIdx+1) * 2_000_000], self.targets_))
  
  def readBatch(self, d, batchIdx):
    del self.data_
    del self.targets_
    self.data_ = np.array([])
    self.targets_ = np.array([], dtype=np.int)
    from glob import glob
    files = glob(os.path.join(d, "*.npy"))
    for f in files:
      if 'data' in f:
        try:
          if f in list(self.seen_files.keys()):
            self.seen_files[f] += 1
            self.read(f, self.seen_files[f])
            self.read('targets{}'.format(f[6:]), self.seen_files[f])
          else:
            batchIdx = 0
            self.read(f, batchIdx)
            self.read('targets{}'.format(f[6:]), batchIdx)
            self.seen_files.update({f: 0})
          break
        except Exception as e:
          print(e)
    print("loaded {} fens and {} results from {}".format(len(self.data_), len(self.targets_), f))
  
  def resetSeenFiles(self):
    self.seen_files = {}

  def toPlanes(self, x):
    PawnKing = np.zeros((4, 64, 1), dtype=np.float)
    QueenRook = np.zeros((4, 64, 1), dtype=np.float)
    KnightBishop = np.zeros((4, 64, 1), dtype=np.float)
    for i in range(8):
      for j in range(8):
        piece = str(x.piece_at(chess.SQUARES[i*8+j]))
        if piece != "None":
          if piece == "P":
            PawnKing[0][i*8+j] = 1
          elif piece == "p":
            PawnKing[1][i*8+j] = 1
          elif piece ==  "K":
            PawnKing[2][i*8+j] = 1
          elif piece ==  "k":
            PawnKing[3][i*8+j] = 1
          
          elif piece ==  "Q":
            QueenRook[0][i*8+j] = 1
          elif piece ==  "q":
            QueenRook[1][i*8+j] = 1
          elif piece ==  "R":
            QueenRook[2][i*8+j] = 1
          elif piece ==  "r":
            QueenRook[3][i*8+j] = 1

          elif piece ==  "N":
            KnightBishop[0][i*8+j] = 1
          elif piece ==  "n":
            KnightBishop[1][i*8+j] = 1
          elif piece ==  "B":
            KnightBishop[2][i*8+j] = 1
          elif piece ==  "b":
            KnightBishop[3][i*8+j] = 1
          else:
            print("we shouldnt get here")
    return PawnKing, QueenRook, KnightBishop

  def prepareData(self):
    """a = 0
    l = []
    first_time = True
    for i in self.data_:
      if np.random.random() > 0.99:
        continue
      a += 1
      if a % 1_000_000 == 999_999 or psutil.virtual_memory().percent > 80:
        st = time()
        if first_time:
          self.trainingData = np.array(l)
          first_time = False
        else:
          self.trainingData = np.concatenate((np.array(l, dtype=np.int), self.trainingData))
        del l
        if time() - st > 60:
          break
        if psutil.virtual_memory().percent > 80:
          break
        l = []
      if a % 10000 == 5000:
        if open("stop.txt").read() == "yes":
          self.trainingData = np.concatenate((np.array(l, dtype=np.int), self.trainingData))
          del l
          break
        print(a+1, end='\r')
      l.append(self.toPlanes(chess.Board(i)))
    targets = []
    for i in range(len(self.targets_)):
      targets.append(0.5 * (self.targets_[i]+1))
    self.targets = np.array(targets, dtype=np.int)
    del targets
    del self.targets_
    #del self.data_
    print("done preparing {} data points".format(len(self.targets)))
    """
  
  def __len__(self):
    assert len(self.data_) == len(self.targets_)
    return len(self.targets_)
  
  def __getitem__(self, idx):
    try:
      return self.toPlanes(chess.Board(self.data_[idx])), (self.targets_[idx]+1) * 0.5
    except:
      self.offset += 1
      return self.toPlanes(chess.Board(self.data_[idx + self.offset])), (self.targets_[idx + self.offset]+1) * 0.5

def match():
  def search(board, color):
    best_s = -2
    best_m = None
    for i in board.legal_moves:
      board.push(i)
      if board.is_game_over():
        if board.result() == "1/2-1/2":
          out = 0.5
        elif board.result() == "1-0" or board.result() == "0-1":
          out = 1.0
      else:
        PawnKing, QueenRook, KnightBishop = net.toPlanes(board)
        PawnKing = torch.from_numpy(PawnKing).view(-1, 4 * 64)
        PawnKing = PawnKing.cuda()
        
        QueenRook = torch.from_numpy(QueenRook).view(-1, 4 * 64)
        QueenRook = QueenRook.cuda()

        KnightBishop = torch.from_numpy(KnightBishop).view(-1, 4 * 64)
        KnightBishop = KnightBishop.cuda()
        if color == 1:
          out = net(PawnKing.float(), QueenRook.float(), KnightBishop.float())
        else:
          out = 1 - net(PawnKing.float(), QueenRook.float(), KnightBishop.float())
      board.pop()
      if out > best_s:
        best_s = out
        best_m = i
      assert best_s != -2, board.fen()
    return best_m, best_s
  board = chess.Board()
  color = 1
  moves = ""
  movesWith = ""
  i = 0
  while not board.is_game_over():
    i += 1
    move, eval = search(board, color)
    moves += str(i) + ". " + str(board.san(move)) + " "
    movesWith += str(i) + ". " + str(board.san(move)) + " { eval: %s }" % float(eval) + " "
    board.push(move)
    color *= -1
    move, eval = search(board, color)
    moves += str(board.san(move)) + " "
    movesWith += str(board.san(move)) + " { eval: %s }" % float(eval) + " "
    board.push(move)
    color *= -1
  print(movesWith)
  open("game.pgn", "w+").write(moves)

def save_parameters():
  print("saving parameters...")
  torch.save(net.state_dict(), "./networks/SUBNETepoch{}.pt".format(epoch))

def test():
  net = Net()
  if len(argv) > 1:
    net.load_state_dict(torch.load(argv[1]))
  net = net.cuda()
  while True:
    i = input()
    x = torch.from_numpy(np.array([net.toPlanes(chess.Board(i))])).cuda()
    out = net(x.float())

    print(float(out[0]))

if __name__ == "__main__":
  #test()
  dataset = DataManager('.')
  #data = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=6)

  net = Net()
  if len(argv) > 1:
    net.load_state_dict(torch.load(argv[1]))
  net = net.cuda()

  #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, 
  #                      nesterov=True, weight_decay=0)
  import ranger
  optimizer = ranger.Ranger(net.parameters())
  
  criterion = torch.nn.L1Loss()
  scheduler = ExponentialLR(optimizer, gamma=0.1)

  import atexit
  def at_exit():
    print('Finished Training')
    save_parameters()
    match()
  atexit.register(at_exit)
  t = 0
  for epoch in range(700):  # loop over the dataset multiple times
    sample_outputs = []
    save_parameters()
    dataset.readBatch(".", t)
    t += 1
    
    try:
      data = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=6)
    except:
      t = 0
      dataset.resetSeenFiles()
      dataset.readBatch(".", 0)
      data = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=6)
    running_loss = 0.0
    print("started epoch {}".format(epoch + 1))
    st = time()
    for i, ((PawnKing, QueenRook, KnightBishop), y) in enumerate(data, 0):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      PawnKing = PawnKing.view(-1, 4 * 64)
      PawnKing = PawnKing.cuda()
      
      QueenRook = QueenRook.view(-1, 4 * 64)
      QueenRook = QueenRook.cuda()

      KnightBishop = KnightBishop.view(-1, 4 * 64)
      KnightBishop = KnightBishop.cuda()
      out = net(PawnKing.float(), QueenRook.float(), KnightBishop.float())
      if i % 200 == 0:
        sample_outputs.append([float(out[0]), float(y[0])])
      loss = criterion(out, torch.from_numpy(np.array(y)).float().view(-1, 1).cuda()) # torch.round(x * 10) / (10)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 400 == 399:    # print every 400 mini-batches
        print('[%d, %5d] loss: %.3f sample outputs: %s' %
            (epoch + 1, i + 1, running_loss / 400, str(sample_outputs)))
        running_loss = 0.0
        sample_outputs = []
    scheduler.step()
  at_exit()