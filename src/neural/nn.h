#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include "neural/simd.h"

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
};