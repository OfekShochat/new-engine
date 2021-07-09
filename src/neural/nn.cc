#include "nn.h"

Layer<768, 512> InferenceNet1::l1;
Layer<512, 1> InferenceNet1::l2;

void InferenceNet1::eval(float* input,
                         float out[1]) {
  float l1out[512];
  l1.eval(input, l1out);
  l2.eval(l1out, out);
}