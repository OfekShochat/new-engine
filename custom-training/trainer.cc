#include "trainer.h"
#include "helper.h"
#include "chess/Position.h"
#include <string>
#include <vector>
#include <fstream>
#include <list>
#include <tuple>
#include <iostream>
#include <cassert>

namespace sunset::training {
Trainer::Trainer(size_t batch_size) {
  batch_size_ = batch_size;
}

void Trainer::ExecuteBatch(std::vector<std::tuple<float*, float>> iter) {
  assert_(iter.size() == batch_size_, "iter.size() has to equal to batch_size_");
  float loss1 = 0.0f;
  float loss2 = 0.0f;
  float loss3 = 0.0f;
  float loss4 = 0.0f;

  for (size_t i = 0; i < iter.size(); i+= 4) {
    {
      auto [input, target] = iter[i + 0];
      float out[1];
      net_.eval(input, out);
      loss1 += BinaryCrossEntropyLossDerivative(target, out[0]);
    }

    {
      auto [input, target] = iter[i + 1];
      float out[1];
      net_.eval(input, out);
      loss2 += BinaryCrossEntropyLossDerivative(target, out[0]);
    }

    {
      auto [input, target] = iter[i + 2];
      float out[1];
      net_.eval(input, out);
      loss3 += BinaryCrossEntropyLossDerivative(target, out[0]);
    }

    {
      auto [input, target] = iter[i + 3];
      float out[1];
      net_.eval(input, out);
      loss4 += BinaryCrossEntropyLossDerivative(target, out[0]);
    }
  }

  const float total = (loss1 + loss2 + loss3 + loss4) / iter.size();
  auto [l1_weights, l1_biases, l2_weights, l2_biases] = net_.parameters();
  std::cout << "l1w" << std::endl;
  optimizer_.step<768 * 512>(l1_weights, total);
  std::cout << "l1b" << std::endl;
  optimizer_.step<512>(l1_biases, total);
  std::cout << "l2w" << std::endl;
  optimizer_.step<512>(l2_weights, total);
  std::cout << "l2b" << std::endl;
  optimizer_.step<1>(l2_biases, total);
}

std::vector<std::tuple<float*, float>> Trainer::GatherBatch(std::string file, size_t index) {
  std::ifstream input_file(file.c_str());
  std::string line;
  for (size_t l_no = 0; l_no < index; l_no++) {
    std::getline(input_file, line);
  }
  assert(input_file.good());
  std::list<std::tuple<float*, float>> positions{};
  for (size_t l_no = 0; l_no < batch_size_; l_no++) {
    std::getline(input_file, line);
    auto splitted = split(line, '|');

    libchess::Position pos(splitted[0]);
    float position_input[64] = {};
    net_.make(pos, position_input);
    
    positions.push_back({ position_input, std::stoi(splitted[1]) - std::stoi(splitted[3]) });
  }
  return std::vector<std::tuple<float*, float>> {positions.begin(), positions.end()};
}

void Trainer::LoadNetFromFile() {
  #include "net.h"
  float dense1_w[512 * 768];
  float dense1_b[512];
  float out_w[512];
  float out_b[1];
  std::copy(dense1_weights, dense1_weights + 768 * 512, dense1_w);
  std::copy(dense1_bias, dense1_bias + 512, dense1_b);
  std::copy(out_weights, out_weights + 512, out_w);
  std::copy(out_bias, out_bias + 1, out_b);

  net_.load(dense1_w, dense1_b,
            out_w, out_b);
}
} // namespace sunset::training