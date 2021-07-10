#pragma once

#include <string>
#include <iostream>

const void assert_(bool x, std::string message) {
  if (!x) {
    std::cout << message << std::endl;
  }
}

std::vector<std::string> split(std::string text, char splitter) {
  std::stringstream ss(text); 
  std::vector<std::string> out{};
 
  std::string s; 
  while (std::getline(ss, s, splitter)) { 
    out.push_back(s); 
  }
  return out;
}