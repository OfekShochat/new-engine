#include "search/search.h"
#include "chess/Position.h"
#include "util/fastmath.h"
#include <mutex>
#include <algorithm>
#include <tuple>
#include <memory>
#include <iostream>

#define MAX_PLY 256
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

int Stack::elapsedTime() {
  std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
}

void Stack::resetTimer() {
  start_time = std::chrono::high_resolution_clock::now();
}

void Stack::incrementNodes() {
  std::lock_guard<std::mutex> lock(NodesMutex);
  nodes++;
}

void Searcher::printThink(std::shared_ptr<Stack> shared, int eval, int depth) {
  int elapsed = shared->elapsedTime();
  std::cout << "info depth " << depth << " score cp " << eval << " nodes " << shared->nodes << " nps " << shared->nodes/elapsed*1000 << " time " << elapsed << std::endl;
}

int Searcher::Quiescence(std::shared_ptr<Stack> shared, libchess::Position pos, int alpha, int beta, int curr_depth) {
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

int Searcher::AlphaBeta(std::shared_ptr<Stack> shared, libchess::Position pos, int alpha, int beta, int curr_depth, int max_depth, Limiter limits, bool PVNode) {
  libchess::MoveList moves = pos.legal_move_list();
  // Position is drawn
  if (pos.is_repeat(3) || pos.halfmoves() >= 100)
    return 0;
  if (moves.size() == 0) {
    if (pos.in_check()) {
      return -MATE + curr_depth;
    }
    return 0;
  }

  if (shared->nodes % 2048 == 0
      && (shared->nodes > limits.nodes
      ||  shared->elapsedTime() > limits.time))
    return static_cast<int>(util::FastLog(0.5/(1-0.5))); // TODO(ghostway): make it actually evaluate

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
    shared->incrementNodes();
    TTHit = true;
    TTE = shared->fromTT(pos.hash());
  }

  if (   TTHit
      && !PVNode
      && TTE.depth >= max_depth)
    return TTE.eval;

  int bestScore = -MATE;
  int moveIdx = 0;
  for (const libchess::Move& m : moves) {
    moveIdx++;
    shared->incrementNodes();
    pos.make_move(m);
    const int score = -AlphaBeta(shared, pos, -beta, -alpha, curr_depth + 1, max_depth, limits);
    pos.unmake_move();
    
    //  if (curr_depth == 0) std::cout << m << " " << score << " " << curr_depth << " " << i << std::endl;

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
    if (moveIdx > 10) {
      TTEntry req;
      req.eval = bestScore;
      req.depth = max_depth;
      shared->addToTT(pos.hash(), req);
    }
  }
  return bestScore;
}

std::tuple<libchess::Move, int> Searcher::SearchPos(std::shared_ptr<Stack> shared, libchess::Position pos, int depth, Limiter limits) {
  shared->resetTimer();
  libchess::Move bestMove;
  int alpha = -MATE;
  int beta = MATE;
  libchess::MoveList moves = pos.legal_move_list();
  int bestScore = -MATE;
  for (const libchess::Move& m : moves) {
    pos.make_move(m);
    const int score = -AlphaBeta(shared, pos, alpha, beta, 0, depth != -1 ? depth : MAX_PLY, limits);
    std::cout << m << " " << score << std::endl;
    pos.unmake_move();

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
      if (score > alpha) {
        alpha = score;
        if (score >= MATE) {
          break;
        }
      }
    }
  }
  printThink(shared, bestScore, depth);
  return { bestMove, bestScore };
}