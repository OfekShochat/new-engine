#pragma once

#include <cmath>
#include <iostream>

namespace sunset::training {
class AdagradOptimizer {
 private:
  float lr_;
  float diagGt;
  float epsilon_;

 public:
  AdagradOptimizer(float lr = 0.001, float epsilon = 1e-7);

  template <int LENGTH>
  void step(float* params, float loss_derivative) {
    diagGt += loss_derivative * loss_derivative;

    //#pragma omp parallel for
    for (int i = 0; i < LENGTH; i++) {
      params[i] = params[i] - lr_ * loss_derivative / std::sqrt(diagGt + epsilon_);
    }
  }
};

AdagradOptimizer::AdagradOptimizer(float lr, float epsilon) {
  lr_ = lr;
  epsilon_ = epsilon;
}

float Sigmoid(float x, float scale) {
  return 1 / (1 + std::exp(-x / scale));
}

float BinaryCrossEntropyLoss(float y_true, float y_pred) {
  return y_true * log10f(Sigmoid(y_pred, 400)) + (1 - y_true) * log10f(1 - Sigmoid(y_pred, 400));
}

float BinaryCrossEntropyLossDerivative(float y_true, float y_pred /*= yhat */) {
  const float y_true_sigmoid = Sigmoid(y_true, 400);
  const float y_pred_sigmoid = Sigmoid(y_pred, 400);
  std::cout << y_true_sigmoid << " " << y_pred_sigmoid << std::endl;
  return -(y_true_sigmoid / y_pred_sigmoid - (1 - y_true_sigmoid) / (1 - y_pred_sigmoid));
}
} // namespace training