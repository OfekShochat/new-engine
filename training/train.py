import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import data
import config_parser

def crossEntropy_loss(x, p):
  return torch.mean(-(p*F.logsigmoid(x) + (1-p)*F.logsigmoid(-x)))

def main():
  cfg = config_parser.Parser("config.json")

  network = model.SAC1(crossEntropy_loss, optim.Adadelta).cuda()
  dataset = data.DataManager(cfg.data_dir)

  for e in range(cfg.epochs):
    dataset.readBatch()
    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.workers)
    for i, (x, y) in enumerate(dataLoader):
      network.train_step(x, y)

if __name__ == "__main__":
  main()