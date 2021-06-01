#include <map> // TODO(ghostway): try this with unordered_map, see if its faster
#include <atomic>
#include <mutex>
#include "chess/Position.h"

struct TTEntry {
  int eval;
  int depth;
};

class Stack {
 private:
  std::mutex TTMutex;
  std::map<int, TTEntry> TT; // TODO(ghostway): try this with unordered_map, see if its faster
  std::mutex KillersMutex;
  std::mutex NodesMutex;
  libchess::Move killers[100][2];
 public:
  //Stack();
  void addToTT(int key, TTEntry entry);
  TTEntry fromTT(int key);
  bool TTContains(int key);
  void UpdateKillers(int depth, libchess::Move m);
  void incrementNodes();

  int nodes = 0;
};

namespace search {
  int AlphaBeta(Stack* shared, libchess::Position pos, int alpha, int beta, int curr_depth, int max_depth, bool PVNode = false);
};