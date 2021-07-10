#pragma once

#include "neural/simd.h"
#include "chess/Position.h"

#include <cstdint>
#include <cstring>

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

class InferenceNet1 {
 private:
  static Layer<768, 512> l1;
  static Layer<512, 1> l2;

 public:
  void load(float* weights1, float* biases1,
            float* weights2, float* biases2) {
    l1.load(weights1, biases1);
    l2.load(weights2, biases2);
  }
  void eval(float* input,
            float out[1]);

  void make(libchess::Position pos,
            float* out) {
    for (int i = 0; i < 64; i++) {
      auto p = pos.piece_on(libchess::Square(i));
      if (p.has_value()) {
        switch (p.value().val()) {
          case libchess::constants::WHITE_PAWN.val():
            out[0 * 64 + i] = 1.0f;
            break;
          case libchess::constants::WHITE_KNIGHT.val():
            out[1 * 64 + i] = 1.0f;
            break;
          case libchess::constants::WHITE_BISHOP.val():
            out[2 * 64 + i] = 1.0f;
            break;
          case libchess::constants::WHITE_ROOK.val():
            out[3 * 64 + i] = 1.0f;
            break;
          case libchess::constants::WHITE_QUEEN.val():
            out[4 * 64 + i] = 1.0f;
            break;
          case libchess::constants::WHITE_KING.val():
            out[5 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_PAWN.val():
            out[6 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_KNIGHT.val():
            out[7 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_BISHOP.val():
            out[8 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_ROOK.val():
            out[9 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_QUEEN.val():
            out[10 * 64 + i] = 1.0f;
            break;
          case libchess::constants::BLACK_KING.val():
            out[11 * 64 + i] = 1.0f;
            break;

          default:
            break;
        }
      }
    }
  }
};