#include "optimizer.h"
#include <cmath>

namespace sunset::training {
AdagradOptimizer::AdagradOptimizer(float lr, float epsilon) {
  lr_ = lr;
  epsilon_ = epsilon;
}
}