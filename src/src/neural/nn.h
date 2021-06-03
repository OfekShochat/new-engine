#include <cstdint>
#include <cstring>
#include <iostream>
#include "neural/simd.h"
/*
 * O - starter
 * B - board representation
 * the letter at the end indicates the output format
 *  - S: sigmoid
*/
// O1S - first version, with a sigmoid output

template <int NEURONS, int INPUTS>
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

template <int L1, int L2>
class O1S {
 private:
  static Layer<L1, 64> H1();
  static Layer<L2, L1> H2();
 public:
  void load(float* weights1, float* biases1, float* weights2, float* biases2) {
    H1.load(weights1, biases1);
    H2.load(weights2, biases2);
  }
  float eval(float* inputs);
};