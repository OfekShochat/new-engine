#pragma once

#include <cmath>
#include <iostream>

namespace sunset::training {
class AdagradOptimizer {
 private:
  float lr_;
  float diagGt = 0.0;
  float epsilon_;

 public:
  AdagradOptimizer(float lr = 0.001, float epsilon = 1e-7);

  /**
   * optimizer step
   * @param param parameters to be updated
   * @param loss_derivative the loss derivative
   * @tparam LENGTH length of the params
   * @returns None
  */
  template <int LENGTH>
  void step(float* params, float loss_derivative) {
    diagGt += loss_derivative * loss_derivative;

    #pragma omp parallel for
    for (int i = 0; i < LENGTH; i++) {
      //if (std::isnan(params[i] - lr_ * loss_derivative / std::sqrt(diagGt + epsilon_))) {
      //  std::cout << loss_derivative << " " << params[i] << std::endl;
      //}
      params[i] = params[i] - lr_ * loss_derivative / std::sqrt(diagGt + epsilon_);
    }
  }
};

inline float Sigmoid(float x, float scale) {
  return 1 / (1 + std::exp(-x / scale));
}

inline float BinaryCrossEntropyLoss(float y_true, float y_pred) {
  return y_true * log10f(Sigmoid(y_pred, 400)) + (1 - y_true) * log10f(1 - Sigmoid(y_pred, 400));
}

inline float BinaryCrossEntropyLossDerivative(float y_true, float y_pred /*= yhat */) {
  const float y_true_sigmoid = Sigmoid(y_true, 400);
  const float y_pred_sigmoid = Sigmoid(y_pred, 400);
  return -(y_true_sigmoid / (y_pred_sigmoid + 0.001) - (1 - y_true_sigmoid) / (1 - y_pred_sigmoid + 0.001));
}
} // namespace sunset::training