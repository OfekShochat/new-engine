#include "neural/nn.h"
#include "tests/helper.h"
#include <iostream>

int main() {
  Layer<16, 1> l;
  float w[16] = {1.0, 0.4f, 0.2f, 0.1f, 0.6f, 0.7f, 0.2f, 0.52f, 0.154f, 0.53456f, 0.74315f, 0.4f, 0.432f, 0.423f, 0.42f, 0.6f};
  float b[1] = {1.0};
  float inputs[16] = { 1.0, 0.4f, 0.2f, 0.1f, 0.6f, 0.7f, 0.2f, 0.52f, 0.154f, 0.53456f, 0.74315f, 0.4f, 0.432f, 0.423f, 0.42f, 0.6f };
  l.load(w, b);
  float o[1];
  l.eval(inputs, o);
  std::cout << o[0] << std::endl;
  return EXPECT<float>(o[0], 5.2941);
}