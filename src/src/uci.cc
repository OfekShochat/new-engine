#include "uci.h"
#include "util/version.h"
#include "util/uciHelper.h"
#include "search/search.h"
#include <iostream>
#include <memory>
#include <string>
#include <iomanip>

void UciHandler::printUci() {
  std::cout << "id name sunset " << VERSION << std::endl;
  std::cout << "id author Ofek Shochat" << std::endl;
  std::cout << "uciok" << std::endl;
}

void UciHandler::isready() {
  shared = std::make_unique<Stack>();
  std::cout << "readyok" << std::endl;
}

void UciHandler::newgame() {
  shared = std::make_unique<Stack>();
}

void UciHandler::loop() {
  bool quit = false;
  while (!quit) {
    std::string cmd;
    std::getline(std::cin, cmd);

    std::string first = cmd.substr(0, cmd.find(" "));
    switch (hash(first.c_str())) {
      case hash("go"):
        go(split(cmd));
        break;
      case hash("quit"):
        quit = true;
        break;
      default:
        std::cout << "invalid command" << std::endl;
    }
  }
}

void UciHandler::go(std::vector<std::string> cmd) {
  int depth = -1;
  int nodes = -1;
  int time = -1;
  bool timemn = false;

  for (int i = 1; i < cmd.size(); i += 1) {
    if (   depth > -1
        || nodes > -1
        || (time  > -1
        && !timemn))
      break;
    switch (hash(cmd[i].c_str())) {
      case hash("depth"):
        if (std::stoi(cmd[i + 1]) < 1) throw std::out_of_range("depth out of range: " + cmd[i + 1]);
        depth = std::stoi(cmd[i + 1]);
        break;
      case hash("nodes"):
        if (std::stoi(cmd[i + 1]) < 1) throw std::out_of_range("nodes out of range: " + cmd[i + 1]);
        nodes = std::stoi(cmd[i + 1]);
        break;
      default:
        throw std::out_of_range("parsing error: " + cmd[i]);
        break;
    }
  }

  Limiter limits;
  limits.nodes = nodes == -1 ? 1000000000 : nodes;
  limits.time  = time ==  -1 ? 1000000000 : time;
  libchess::Position b(fen);
  searcher.SearchPos(shared, b, depth > -1 ? 100 : depth, limits);
}