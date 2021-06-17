import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    self.optimizer = optimizer(self.parameters())

    self.pbar = tqdm(total=10000)

  def forward(self, PawnKing, QueenRook, KnightBishop):
    PawnKing     = self.SubPawnKing_dense(PawnKing)
    QueenRook    = self.SubQueenRook_dense(QueenRook)
    KnightBishop = self.SubKnightBishop_dense(KnightBishop)
    added = torch.add(PawnKing, QueenRook)
    added = torch.add(added, KnightBishop)

    x = self.dropout1(added)
    x  = self.MainInput_dense(added)
    x = self.dropout2(x)
    x  = self.MainOut_dense(x)
    return x
  
  def train_step(self, data: tuple, target: float) -> None:
    PawnKing = data[0].view(-1, 4 * 64)
    PawnKing = PawnKing.cuda()
    
    QueenRook = data[1].view(-1, 4 * 64)
    QueenRook = QueenRook.cuda()

    KnightBishop = data[2].view(-1, 4 * 64)
    KnightBishop = KnightBishop.cuda()
    out = self(PawnKing.float(), QueenRook.float(), KnightBishop.float())
    
    loss = self.loss(out, target.float().view(-1, 1).cuda())
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    self.log(loss.item())

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

  def log(self, loss):
    self.pbar.update()
    self.pbar.set_description("loss: {}".format(loss), refresh=False)