#include "seq/example/include/ops_seq.hpp"
#include <functional>
#include <random>
#include <thread>

using namespace std::chrono_literals;

std::string volochaev_s_count_characters_27_seq::get_random_string(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::string vec(sz, ' ');
  for (size_t i = 0; i < sz; i++) {
    vec[i] += gen() % 256;
  }
  return vec;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::pre_processing() {
  internal_order_test();
  // Init value for input and output
  std::string input1_ = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
  std::string input2_ = reinterpret_cast<std::string*>(taskData->inputs[0])[1];

  input_ = std::vector<std::pair<char, char>>(std::min(input1_.size(), input2_.size()));

  for (size_t i = 0; i < std::min(input1_.size(), input2_.size()); ++i) {
    input_[i].first = input1_[i];
    input_[i].second = input2_[i];
  }

  sz1 = input1_.size();
  sz2 = input2_.size();
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::run() {
  internal_order_test();

  res = abs(sz1 - sz2);

  for (auto [x, y] : input_) {
    if (x != y) {
      res += 2;
    }
  }
  
  return true;
}

bool volochaev_s_count_characters_27_seq::Lab1_27::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}
