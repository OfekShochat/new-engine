import json
from multiprocessing import cpu_count

class Parser:
  def __init__(self, cfg_file) -> None:
    self.parse(cfg_file)

  def parse(self, cfg_file):
    cfg = json.loads(open(cfg_file).read())
    cfg.setdefault("data_dir", "./data")
    cfg.setdefault("batch_size", 1024)
    cfg.setdefault("workers", cpu_count() - 2)
    cfg.setdefault("epochs", 100)

    self.batch_size = cfg["batch_size"]
    self.workers = cfg["workers"]
    self.epochs = cfg["epochs"]
    self.data_dir = cfg["data_dir"]