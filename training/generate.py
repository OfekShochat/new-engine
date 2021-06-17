from tqdm import tqdm
from io import StringIO
import atexit
from time import sleep
import signal
#from dataset import DataManager
import numpy as np

#datamn = DataManager()

def from_pgn():
  import chess.pgn
  datapoints = 0
  pgn = open("/home/ghostway/projects/cpp/sunset/training/data/lichess_elite_2021-04.pgn")
  splitted_pgn = pgn.read().split("\n\n")

  def at_exit():
    datamn.data_ = np.array(fens)
    datamn.targets_ = np.array(targets)
    datamn.write()
    print("{} average moves per game".format(datapoints/games))
  atexit.register(at_exit)

  signal.signal(signal.SIGTERM, at_exit)
  signal.signal(signal.SIGINT, at_exit)

  targets = []
  fens = []
  games = 0
  pbar = tqdm(splitted_pgn, unit="games")
  for g in pbar:
    if games % 100000 == 999_999:
      at_exit()
    games += 1
    game = chess.pgn.read_game(StringIO(g))

    board = game.board()
    result = game.headers["Result"]
    if result == "1-0":
      r = 1
    elif result == "0-1":
      r = 0
    elif result == "1/2-1/2":
      r = 0.5

    for move in game.mainline_moves():
      datapoints += 1
      fens.append(board.fen())
      targets.append(r)
      board.push(move)

def from_random_engine_eval():
  import chess
  import chess.engine

  stages = [20, 400, 1000, 2000, 5000, 10000, 20000]
  def random_fen(move_num):
    b = chess.Board()
    for _ in range(move_num):
      b.push(np.random.choice(tuple(b.generate_legal_moves())))
      if b.is_game_over():
        b.pop()
        break
    return b

  engine = chess.engine.SimpleEngine.popen_uci("/home/ghostway/projects/cpp/realStockfish2/src/stockfish")
  engine.configure({"Threads": 16})

  targets = []
  fens = []

  def at_exit():
    datamn.data_ = np.array(fens)
    datamn.targets_ = np.array(targets, dtype=np.float16)
    datamn.write()
    engine.close()
  atexit.register(at_exit)

  m_n = 1
  samples_per_moveStage = 20
  pbar = tqdm(range(100_000_000), unit="pos")
  for i in pbar:
    #if i % samples_per_moveStage == 0:
    #  m_n += 1
    #  samples_per_moveStage = stages[m_n - 1] if m_n-1 < len(stages) else stages[-1] * m_n//2
    
    b = random_fen(np.random.randint(1, 40))
    e = engine.analyse(b, chess.engine.Limit(depth=1))

    fens.append(b.fen())
    targets.append(e["score"].white().wdl().expectation())

def from_pgn_engine_eval(threadIdx):
  import chess
  import chess.engine
  import chess.pgn
  datapoints = 0
  pgn = open("data/1.pgn")
  splitted_pgn = pgn.read().split("\n\n")[threadIdx * 100_000: (threadIdx+1) * 100_000]

  engine = chess.engine.SimpleEngine.popen_uci("/home/ghostway/projects/cpp/realStockfish2/src/stockfish")
  engine.configure({"Threads": 2})

  targets = []
  fens = []
  games = 0

  def at_exit():
    datamn.data_ = np.array(fens)
    datamn.targets_ = np.array(targets, dtype=np.float16)
    datamn.write()
    engine.close()
  atexit.register(at_exit)

  pbar = tqdm(splitted_pgn, unit="games")
  for g in pbar:
    if games % 15_000 == 15_000-1:
      pbar.set_description("sleeping")
      sleep(2)
      pbar.set_description("")
    games += 1
    game = chess.pgn.read_game(StringIO(g))

    board = game.board()

    for move in game.mainline_moves():
      datapoints += 1
      fens.append(board.fen())
      e = engine.analyse(board, chess.engine.Limit(depth=1))
      targets.append(e["score"].white().wdl().expectation())
      board.push(move)

def from_seer_data(root):
  from glob import glob

  fens = []
  targets = []

  def at_exit():
    datamn.data_ = np.array(fens)
    datamn.targets_ = np.array(targets, dtype=np.float16)
    datamn.write()
  atexit.register(at_exit)

  files = glob("{}/*.txt".format(root))[17:]
  for f in files:
    print(f)
    fens = []
    targets = []
    for d in open(f).readlines():
      s = d.split("|")
      fens.append(s[0])
      try: targets.append(int(s[1])/1024 + 0.5 * (int(s[2])/1024))
      except ValueError:
        fens.pop()
    datamn.data_ = np.array(fens)
    datamn.targets_ = np.array(targets, dtype=np.float16)
    datamn.write()

def concatenate():
  from glob import glob
  files = glob("*.npy")
  files.sort()
  data = np.array([])
  targets = np.array([])
  for f in files:
    try:
      if 'data' in f:
        if len(data) == 0:
          data = np.load(f)
        data = np.concatenate(np.array(np.load(f, allow_pickle = False)), data)
      else:
        if len(targets) == 0:
          targets = np.load(f)
        targets = np.concatenate(np.array(np.load(f, allow_pickle = False)), targets)
    except Exception as e:
      print(e, f)

  np.save("data", data)
  np.save("targets", targets)

if __name__ == "__main__":
  concatenate()