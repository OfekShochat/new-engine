#pragma once

#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>

std::vector<std::string> split(std::string text) {
  char space_char = ' ';
  std::vector<std::string> words{};

  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, space_char)){
    word.erase(std::remove_if(word.begin(), word.end(), ispunct), word.end());
    words.push_back(word);
  }
  return words;
}

uint64_t constexpr mix(char m, uint64_t s) {
  return ((s<<7) + ~(s>>3)) + ~m;
}

uint64_t constexpr hash(const char * m) {
  return (*m) ? mix(*m,hash(m+1)) : 0;
}