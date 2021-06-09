#include "neural/nn.h"

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