#pragma once

#include "nn.h"
#include "optimizer.h"
#include <cstring>
#include <vector>
#include <tuple>
#include <string>

namespace sunset::training {
class Trainer {
 private:
  size_t batch_size_;
  Inference::InferenceNet1 net_;
  AdagradOptimizer optimizer_;
 public:
  Trainer(size_t batch_size);
  void ExecuteBatch(std::vector<std::tuple<float*, float>> iter);
  std::vector<std::tuple<float*, float>> GatherBatch(std::string file, size_t index);
  void LoadNetFromFile();
};
} // namespace sunset::training