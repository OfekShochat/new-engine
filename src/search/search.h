#pragma once

#include "chess/Position.h"
#include <map> // TODO(ghostway): try this with unordered_map, see if its faster
#include <mutex>
#include <tuple>
#include <memory>
#include <chrono>

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

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
 public:
  //Stack();
  void addToTT(int key, TTEntry entry);
  TTEntry fromTT(int key);
  bool TTContains(int key);
  void UpdateKillers(int depth, libchess::Move m);
  void incrementNodes();
  void resetTimer();
  int elapsedTime();

  int nodes = 0;
};

struct Limiter {
  int nodes;
  int time;
};

class Searcher {
 private:
  void printThink(std::shared_ptr<Stack> shared, int eval, int depth);
  int Quiescence(std::shared_ptr<Stack> shared, libchess::Position pos, int alpha, int beta, int curr_depth);
  int AlphaBeta(std::shared_ptr<Stack> shared, libchess::Position pos, int alpha, int beta, int curr_depth, int max_depth, Limiter limits, bool PVNode = false);
 public:
  std::tuple<libchess::Move, int> SearchPos(std::shared_ptr<Stack> shared, libchess::Position pos, int depth, Limiter limits);
};