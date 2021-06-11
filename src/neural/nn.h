#pragma once

#include "neural/simd.h"
#include "chess/Position.h"

#include <cstdint>
#include <cstring>
#include <iostream>

template <int INPUTS, int NEURONS>
class Layer {
 private:
  alignas(alignment) float weights_[NEURONS * INPUTS];
  alignas(alignment) float biases_[NEURONS];
  alignas(alignment) float inputs_[INPUTS];
 public:
  void load(float weights[NEURONS * INPUTS], float biases[NEURONS]) {
    std::memcpy(weights_, weights, sizeof(float) * NEURONS * INPUTS);
    std::memcpy(biases_, biases, sizeof(float) * NEURONS);
  }

  void eval(float inputs[INPUTS], float out[NEURONS]) {
    std::memcpy(inputs_, inputs, sizeof(float) * INPUTS);
    for (int i = 0; i < NEURONS; i++) {
      out[i] = biases_[i] + dot_product<INPUTS>(weights_ + i * INPUTS, inputs_);
    }
  }
};

class SAC1 {
 private:
  static Layer<4 * 64, 64> SubPawnKing;
  static Layer<4 * 64, 64> SubQueenRook;
  static Layer<4 * 64, 64> SubKnightBishop;

  static Layer<64, 32> MainInput;
  static Layer<32, 1> MainOut;
 public:
  void load(float* weights1, float* biases1,
            float* weights2, float* biases2,
            float* weights3, float* biases3,
            float* weights4, float* biases4,
            float* weights5, float* biases5) {
    SubPawnKing.load(weights1, biases1);
    SubQueenRook.load(weights2, biases2);
    SubKnightBishop.load(weights3, biases3);
    
    MainInput.load(weights4, biases4);
    MainOut.load(weights5, biases5);
  }
  void eval(float* PawnKing, bool skipPawnKing,
            float* KnightBishop, bool skipKnightBishop,
            float* QueenRook, bool skipQueenRook,
            float out[1]);
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
  return {true, true, true}; // check hash of pawnsKing, QueenRook, and KnightBishop to the cache hash.
  }
};