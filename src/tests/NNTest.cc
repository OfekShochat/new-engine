#include "neural/nn.h"
#include "tests/helper.h"
#include "chess/Position.h"
#include <iostream>

int main() {
  srand (time(NULL));
  InferenceNet1 NN;

  libchess::Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  float input[768];
  NN.make(pos, input);

  NN.load(*RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<256 * 64>(), *RandArray<64>());
  float out[1];
  NN.eval(input, out);

  std::cout << out[0] << std::endl;
  return 0;
}