#include "optimizer.h"
#include <iostream>

int main() {
  std::cout << "poop" << std::endl;
  sunset::training::AdagradOptimizer optimizer;
  float params[5] = { 1.0f,2.0f,3.0f,4.0f,5.0f};
  optimizer.step<5>(params, sunset::training::BinaryCrossEntropyLossDerivative(0.1, 0.2));
  std::cout << params[0] << std::endl;
  return 0;
}