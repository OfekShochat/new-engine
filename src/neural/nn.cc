#include "neural/nn.h"
#include "chess/Position.h"

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

std::tuple<bool, bool, bool> make(libchess::Position pos,
                                  float* PawnKing,
                                  float* KnightBishop,
                                  float* QueenRook) {
  for (int i = 0; i < 64; i++) {
    auto p = pos.piece_on(libchess::Square(i));
    if (p.has_value()) {
      switch (p.value().val()) {
        case libchess::constants::WHITE_PAWN.val():
          PawnKing[0 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_PAWN.val():
          PawnKing[1 * 64 + i] = 1.0;
          break;
        case libchess::constants::WHITE_KING.val():
          PawnKing[2 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_KING.val():
          PawnKing[3 * 64 + i] = 1.0;
          break;
        
        case libchess::constants::WHITE_QUEEN.val():
          QueenRook[0 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_QUEEN.val():
          QueenRook[1 * 64 + i] = 1.0;
          break;
        case libchess::constants::WHITE_ROOK.val():
          QueenRook[2 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_ROOK.val():
          QueenRook[3 * 64 + i] = 1.0;
          break;
        
        case libchess::constants::WHITE_KNIGHT.val():
          KnightBishop[0 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_KNIGHT.val():
          KnightBishop[1 * 64 + i] = 1.0;
          break;
        case libchess::constants::WHITE_BISHOP.val():
          KnightBishop[2 * 64 + i] = 1.0;
          break;
        case libchess::constants::BLACK_BISHOP.val():
          KnightBishop[3 * 64 + i] = 1.0;
          break;

        default:
          break;
      }
    }
  }
  return {false, false, false}; // check hash of pawnsKing, QueenRook, and KnightBishop to the cache hash.
}