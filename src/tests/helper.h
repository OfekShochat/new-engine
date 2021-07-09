#pragma once

#include <cstdlib>
#include <ctime>

constexpr float EPSILON = 0.0001;

bool compare_floats_(float a, float b) {
  float diff = a - b;
  return (diff < EPSILON) && (diff > -EPSILON);
}

template <typename T>
int EXPECT(T what, T expected) {
  switch (sizeof(T)) {
    case (4): // float
      return !compare_floats_(what, expected);
      break;
    case (8): // double
      return !compare_floats_((float)what, (float)expected); // we are saying the opposite of it
                                                            // because if it fails, we want to
                                                            // return 1, to indicate a fail.
      break;
    default:
      return !(what == expected);
  }
}

template <int LENGTH>
float** RandArray() {
  float** array;
  array = (float**)malloc(sizeof(float*) * LENGTH + 1);
  
  for (int i = 0; i < LENGTH; i++) {
    array[i] = (float*)malloc(sizeof(float));
    array[i][0] = 0;
  }

  for (int i = 0; i < LENGTH; i++)
    *(array[i]) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  
  return array;
}