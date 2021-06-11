#include "neural/nn.h"
#include "tests/helper.h"
#include "chess/Position.h"
#include <iostream>

int main() {
  srand (time(NULL));
  SAC1 NN;

  libchess::Position pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  
  float PawnKing[64 * 4];
  float KnightBishop[64 * 4];
  float QueenRook[64 * 4];
  NN.make(pos, PawnKing, KnightBishop, QueenRook);

  NN.load(*RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<64 * 32>(), *RandArray<32>(),
          *RandArray<32 * 1>(), *RandArray<1>());
  float out[1];
  NN.eval(PawnKing, false, KnightBishop, false, QueenRook, false, out);

  std::cout << out[0] << std::endl;
  return 0; //EXPECT<float>(o[0], 5.2941);
}