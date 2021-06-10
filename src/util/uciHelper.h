#pragma once

#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>

std::vector<std::string> split(std::string text) {
  std::stringstream ss(text); 
  std::vector<std::string> out{};
 
  std::string s; 
  while (std::getline(ss, s, ' ')) { 
    out.push_back(s); 
  }
  return out;
}

uint64_t constexpr mix(char m, uint64_t s) {
  return ((s<<7) + ~(s>>3)) + ~m;
}

uint64_t constexpr hash(const char * m) {
  return (*m) ? mix(*m,hash(m+1)) : 0;
}