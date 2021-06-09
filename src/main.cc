#include "search/search.h"
#include "chess/Position.h"
#include "uci.h"
#include <tuple>
#include <iostream>
#include <chrono>

int main(int argc, char *argv[]) {
  UciHandler uci;
  uci.loop();
  return 0;
}