#include "seq/chernova_n_word_count/include/ops_seq.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

std::vector<char> chernova_n_word_count_seq::clean_string(const std::vector<char>& input) {
  std::string result;
  std::string str(input.begin(), input.end());

  std::string::size_type pos = 0;
  while ((pos = str.find("  ", pos)) != std::string::npos) {
    str.erase(pos, 1);
  }

  pos = 0;
  while ((pos = str.find(" - ", pos)) != std::string::npos) {
    str.erase(pos, 2);
  }

  pos = 0;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  pos = str.size() - 1;
  if (str[pos] == ' ') {
    str.erase(pos, 1);
  }

  result.assign(str.begin(), str.end());
  return std::vector<char>(result.begin(), result.end());
}

bool chernova_n_word_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  spaceCount = 0;
  auto* tmp = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  input_ = clean_string(input_);
  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool chernova_n_word_count_seq::TestTaskSequential::run() {
  internal_order_test();
  if (input_.empty()) {
    spaceCount = -1;
  }
  for (size_t i = 0; i < input_.size(); i++) {
    char c = input_[i];
    if (c == ' ') {
      spaceCount++;
    }
  }
  return true;
}

bool chernova_n_word_count_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = spaceCount + 1;
  return true;
}
