#include "neural/nn.h"

template <int L1, int L2>
float O1S<L1, L2>::eval(float* inputs) {
  float L1out = H1.eval(inputs);
  float out = H2.eval(L1out);
  return out;
}