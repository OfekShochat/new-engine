import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import data
import config_parser
from tqdm import tqdm

def log(pbar, loss):
  pbar.set_description("loss: {}".format(loss), refresh=False)
  pbar.update()

def log_last(pbar, loss):
  pbar.set_description("loss: {}".format(loss), refresh=True)

def crossEntropy_loss(x, p):
  return torch.mean(-(p*F.logsigmoid(x) + (1-p)*F.logsigmoid(-x)))

def main():
  cfg = config_parser.Parser("config.json")

  network = model.SAC1(crossEntropy_loss, optim.Adam).cuda()
  dataset = data.DataManager(cfg.data_dir)

  import atexit
  def at_exit():
    network.save_checkpoint("SAC{}.cpkt".format(e), losses/i, e)
  atexit.register(at_exit)

  for e in range(cfg.epochs):
    dataset.readBatch()
    pbar = tqdm(total=len(dataset)//cfg.batch_size)
    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.workers)
    losses = 0.0
    for i, (x, y) in enumerate(dataLoader):
      loss = network.train_step(x, y)
      log(pbar, loss)
      losses += abs(loss)
    log_last(pbar, losses/i)
  network.test()

if __name__ == "__main__":
  main()