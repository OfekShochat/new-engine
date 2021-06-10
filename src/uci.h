#pragma once

#include "chess/Position.h"
#include "search/search.h"
#include <memory>

class UciHandler {
 private:
  // searcher data
  Searcher searcher;
  std::shared_ptr<Stack> shared = std::make_shared<Stack>();
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  void printUci();
  void isready();
  void newgame();
  void go(std::vector<std::string> cmd);
  void position(std::vector<std::string> cmd);
 public:
  void loop();
};