// Copyright 2024 Nesterov Alexander
#include "seq/makhov_m_num_of_diff_elements_in_two_str/include/ops_seq.hpp"

int makhov_m_num_of_diff_elements_in_two_str_seq::countDiffElem(const std::string &str1_, const std::string &str2_) {
  int count = 0;
  int sizeDiff = std::abs(((int)str1_.size() - (int)str2_.size()));
  size_t minSize = std::min(str1_.size(), str2_.size());
  for (size_t i = 0; i < minSize; i++) {
    if (str1_[i] != str2_[i]) count++;
  }
  return count + sizeDiff;
}

bool makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output and strings size
  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->outputs_count[0] == 1;
}

bool makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  str1 = std::string(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  str2 = std::string(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
  res = 0;
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential::run() {
  internal_order_test();
  res = countDiffElem(str1, str2);
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}
