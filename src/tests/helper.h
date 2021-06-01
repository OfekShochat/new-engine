#define EPSILON 0.0001

bool compare_floats_(float a, float b) {
  float diff = a - b;
  return (diff < EPSILON) && (diff > -EPSILON);
}

template<typename T>
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