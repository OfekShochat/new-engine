import torch
import torch.nn as nn
import torch.nn.functional as F
import protopy.SAC1_pb2 as pbsunset
from numpy import array, float32

def entropy(loss, p):
  return loss + (p * (p + 1e-12).log() + (1.0 - p) * (1.0 - p + 1e-12).log())

class SAC1(nn.Module):
  def __init__(self, loss, optimizer):
    super(SAC1, self).__init__()
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
    self.MainInput_dense  = nn.Linear(64, 32)
    self.MainOut_dense    = nn.Linear(32, 1)

    self.dropout1 = nn.Dropout(p = 0.5)
    self.dropout2 = nn.Dropout(p = 0.8)

    self.loss = loss
    self.optimizer = optimizer(self.parameters(), lr=0.0015)

    self.pb = pbsunset.Net()

  def forward(self, PawnKing, QueenRook, KnightBishop): #d
    PawnKing     = F.relu(self.SubPawnKing_dense(PawnKing))
    QueenRook    = F.relu(self.SubQueenRook_dense(QueenRook))
    KnightBishop = F.relu(self.SubKnightBishop_dense(KnightBishop))
    added = torch.add(PawnKing, QueenRook)
    added = torch.add(added, KnightBishop)

    #x = self.dropout1(added)
    x  = F.relu(self.MainInput_dense(added)) #KnightBishop
    #x = self.dropout2(x)
    x  = self.MainOut_dense(x)
    return x
  
  def train_step(self, data: tuple, target: float) -> None:
    PawnKing = data[0].view(-1, 4 * 64)
    PawnKing = PawnKing.cuda()
    
    QueenRook = data[1].view(-1, 4 * 64)
    QueenRook = QueenRook.cuda()

    KnightBishop = data[2].view(-1, 4 * 64)
    KnightBishop = KnightBishop.cuda()
    out = self(PawnKing.float(), QueenRook.float(), KnightBishop.float()) #
    loss = self.loss(out, target.float().view(-1, 1).cuda())
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return float(entropy(loss.item(), target.float().view(-1, 1)[0]))

  def test_step(self, data: tuple, target: float) -> None:
    with torch.no_grad():
      PawnKing = data[0].view(-1, 4 * 64)
      PawnKing = PawnKing.cuda()
      
      QueenRook = data[1].view(-1, 4 * 64)
      QueenRook = QueenRook.cuda()

      KnightBishop = data[2].view(-1, 4 * 64)
      KnightBishop = KnightBishop.cuda()
      out = self(PawnKing.float(), QueenRook.float(), KnightBishop.float())
      
      loss = self.loss(out, target.float().view(-1, 1).cuda())
      self.log(loss.item())

  def toPlanes(self, x):
    PawnKing     = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]
    QueenRook    = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]
    KnightBishop = [[0] * 64, [0] * 64, [0] * 64, [0] * 64]

    i = 0
    if "w" in x:
      color = 1
      #iterable = reversed(x[:x.find(" ")])
    else:
      #iterable = x[:x.find(" ")]
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
    #assert color != 0
    return (torch.FloatTensor(PawnKing) , torch.FloatTensor(QueenRook), torch.FloatTensor(KnightBishop)), color

  def test(self):
    self.eval()
    while True:
      i = input()
      data = self.toPlanes(i)[0]
      PawnKing = data[0].view(-1, 4 * 64)
      PawnKing = PawnKing.cuda()
      
      QueenRook = data[1].view(-1, 4 * 64)
      QueenRook = QueenRook.cuda()

      KnightBishop = data[2].view(-1, 4 * 64)
      KnightBishop = KnightBishop.cuda()
      out = torch.sigmoid(self(PawnKing.float(), QueenRook.float(), KnightBishop.float()))
      print(out)

  def load_checkpoint(self, f):
    checkpoint = torch.load(f)
    self.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.train()
    return checkpoint['epoch']

  def save_checkpoint(self, f, loss, epoch):
    from torch import save
    save({
      'epoch': epoch,
      'model_state_dict': self.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'loss': loss,
    }, f)

  def save_proto(self, f):
    for name, content in self.network.named_parameters():
      exec("self.pb.params.{} = {}".format(name, bytes(content.detach().numpy())))
    data = self.pb.SerializeToString()
    open(f, 'wb+').write(data)