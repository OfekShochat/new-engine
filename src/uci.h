#pragma once

#include "chess/Position.h"
#include "search/search.h"
#include <memory>

class UciHandler {
 private:
  // searcher data
  Searcher searcher;
  std::shared_ptr<Stack> shared = std::make_shared<Stack>();
  std::string fen = "r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/2B2PPP/RN2R1K1 w - - 1 1";

  void printUci();
  void isready();
  void newgame();
  void go(std::vector<std::string> cmd);
 public:
  void loop();
};