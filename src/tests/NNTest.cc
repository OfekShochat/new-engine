#include "neural/nn.h"
#include "tests/helper.h"
#include <iostream>

int main() {
  srand (time(NULL));
  SAC1 NN;
  NN.load(*RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<256 * 64>(), *RandArray<64>(),
          *RandArray<64 * 32>(), *RandArray<32>(),
          *RandArray<32 * 1>(), *RandArray<1>());
  float out[1];
  NN.eval(*RandArray<256>(), false, *RandArray<256>(), false, *RandArray<256>(), false, out);

  std::cout << out[0] << std::endl;
  return 0; //EXPECT<float>(o[0], 5.2941);
}