#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace util {
inline float FastLog2(const float a) {
  uint32_t tmp;
  std::memcpy(&tmp, &a, sizeof(float));
  uint32_t expb = tmp >> 23;
  tmp = (tmp & 0x7fffff) | (0x7f << 23);
  float out;
  std::memcpy(&out, &tmp, sizeof(float));
  out -= 1.0f;
  // Minimize max absolute error.
  return out * (1.3465552f - 0.34655523f * out) - 127 + expb;
}

inline float FastLog(const float a) {
  return 0.6931471805599453f * FastLog2(a);
}
} //  namespace utils