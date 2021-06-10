#include "neural/nn.h"

Layer<4 * 64, 64> SAC1::SubPawnKing;
Layer<4 * 64, 64> SAC1::SubQueenRook;
Layer<4 * 64, 64> SAC1::SubKnightBishop;
Layer<64, 32> SAC1::MainInput;
Layer<32, 1> SAC1::MainOut;

void SAC1::eval(float* PawnKing, bool skipPawnKing,
                 float* KnightBishop, bool skipKnightBishop,
                 float* QueenRook, bool skipQueenRook,
                 float out[1]) {
  float SubKnightBishopOut[64];
  float SubPawnKingOut[64];
  float SubQueenRookOut[64];
  if (!skipPawnKing) SubPawnKing.eval(PawnKing, SubPawnKingOut);
  if (!skipKnightBishop) SubQueenRook.eval(QueenRook, SubQueenRookOut);
  if (!skipQueenRook) SubKnightBishop.eval(KnightBishop, SubKnightBishopOut);

  float SubOut[64];
  for (int i = 0; i < 64; i++)
    SubOut[i] = SubPawnKingOut[i] + SubQueenRookOut[i] + SubKnightBishopOut[i];

  float MainInputOut[32];
  MainInput.eval(SubOut, MainInputOut);
  MainOut.eval(MainInputOut, out);
}