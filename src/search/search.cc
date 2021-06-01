#include "search/search.h"
#include "chess/Position.h"
#include <mutex>
#include <algorithm>
#include "util/fastmath.h"

#include <iostream>

#define MAX_PLY 100
#define MATE 10000
#define MAX_QUIESCENCE_PLY 10

void Stack::addToTT(int key, TTEntry entry) {
  std::lock_guard<std::mutex> lock(TTMutex);
  TT[key] = entry;
}

TTEntry Stack::fromTT(int key) {
  std::lock_guard<std::mutex> lock(TTMutex);
  return TT[key];
}

bool Stack::TTContains(int key) {
  std::lock_guard<std::mutex> lock(TTMutex);
  return TT.count( key ) != 0;
}

void Stack::UpdateKillers(int depth, libchess::Move m) {
  std::lock_guard<std::mutex> lock(KillersMutex);
  if (!(killers[depth][0] == m)) {
    killers[depth][1] = killers[depth][0];
    killers[depth][0] = m;
  }
}

void Stack::incrementNodes() {
  std::lock_guard<std::mutex> lock(NodesMutex);
  nodes++;
}

namespace search {
int Quiescence(Stack* shared, libchess::Position pos, int alpha, int beta, int curr_depth) {
  int eval = static_cast<int>(util::FastLog(0.5/(1-0.5))); // TODO(ghostway): make it actually evaluate
  if (curr_depth > MAX_QUIESCENCE_PLY)
    return eval;
  
  if (eval >= beta)
    return eval;
  
  if (eval > alpha)
    alpha = eval;

  int bestScore = eval;
  libchess::MoveList moves;
  pos.generate_capture_moves(moves, pos.side_to_move());
  for (const libchess::Move& m : moves) {
    pos.make_move(m);
    const int score = -Quiescence(shared, pos, -beta, -alpha, curr_depth + 1);

    if (score > bestScore) {
      bestScore = score;
      if (score > alpha) {
        alpha = score;
        if (score >= beta)
          break;
      }
    }
    pos.unmake_move();
  }
  return bestScore;
}

int AlphaBeta(Stack* shared, libchess::Position pos, int alpha, int beta, int curr_depth, int max_depth, bool PVNode) {
  libchess::MoveList moves = pos.legal_move_list();
  // Position is drawn
  if (pos.is_repeat(3) || pos.halfmoves() >= 100)
    return 0;
  if (moves.size() == 0) {
    if (pos.in_check()) {
      if (pos.side_to_move() == libchess::constants::BLACK)
        return MATE;
      else
        return -MATE;
    }
    return 0;
  }

  // Max depth reached
  if (curr_depth >= MAX_PLY)
    return static_cast<int>(util::FastLog(0.5/(1-0.5))); // TODO(ghostway): make it actually evaluate

  // Mate distance pruning
  alpha = std::max(alpha, -MATE + curr_depth);
  beta  = std::min(beta,   MATE - curr_depth - 1);
  if (alpha >= beta)
    return alpha;

  //if (pos.in_check())
  //  max_depth++;
  
  if (curr_depth >= max_depth)
    return 0; //Quiescence(shared, pos, alpha, beta, 0); // TODO(ghostway): make it actually evaluate or Quiescence search

  bool TTHit = false;
  TTEntry TTE;
  if (shared->TTContains(pos.hash())) {
    TTHit = true;
    TTE = shared->fromTT(pos.hash());
  }

  if (   TTHit
      && !PVNode
      && TTE.depth >= max_depth)
    return TTE.eval;

  int bestScore = -MATE;
  int i = 0;
  for (const libchess::Move& m : moves) {
    shared->incrementNodes();
    pos.make_move(m);
    const int score = -AlphaBeta(shared, pos, -beta, -alpha, curr_depth + 1, max_depth);
    pos.unmake_move();
    
    if (curr_depth == 0) std::cout << m << " " << score << " " << curr_depth << " " << i << std::endl;

    if (score > bestScore) {
      bestScore = score;
      if (score > alpha) {
        alpha = score;
        if (score >= beta) {
          shared->UpdateKillers(curr_depth, m);
          return score;
        }
      }
    }
  }

  TTEntry e;
  e.eval = bestScore;
  e.depth = max_depth;
  shared->addToTT(pos.hash(), e);
  return bestScore;
}
} // namespace  search