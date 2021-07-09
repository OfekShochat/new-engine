#include "tests/helper.h"
#include "search/search.h"
#include <iostream>

int main() {
  Stack d;
  TTEntry e{};
  e.eval = 3;
  d.AddToTT(432, e);

  return EXPECT<bool>((d.FromTT(432).eval == 3
                      && d.TTContains(432)
                      && !d.TTContains(42)),
                      true);
}