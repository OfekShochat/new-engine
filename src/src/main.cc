#include "search/search.h"
#include "chess/Position.h"
#include "uci.h"
#include <tuple>
#include <iostream>
#include <chrono>

int main() {
  libchess::Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  UciHandler uci;
  uci.loop();
  return 0;
}