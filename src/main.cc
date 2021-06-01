#include "search/search.h"
#include <iostream>
#include "chess/Position.h"

int main() {
  libchess::Position pos("6k1/3b3r/1p1p4/p1n2p2/1PPNp3/P3qBp1/1R1R2P1/5K2 w - - 0 3");
  Stack* s = new Stack();
  int v = search::AlphaBeta(s, pos, -10000, 10000, 0, 6);
  std::cout << v << std::endl;
  std::cout << s->nodes << std::endl;
  return 0;
}