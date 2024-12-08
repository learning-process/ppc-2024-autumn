// Copyright 2024 Nesterov Alexander
#include "seq/rezantseva_a_vector_dot_product/include/ops_seq.hpp"

bool rezantseva_a_vector_dot_product_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->outputs.size() == taskData->outputs_count.size()) && taskData->outputs.size() == 1 &&
         taskData->outputs_count[0] == 1;
}

bool rezantseva_a_vector_dot_product_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  input_ = std::vector<std::vector<int>>(taskData->inputs.size());
  for (size_t i = 0; i < input_.size(); i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    input_[i] = std::vector<int>(taskData->inputs_count[i]);
    for (size_t j = 0; j < taskData->inputs_count[i]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res = 0;
  return true;
}

bool rezantseva_a_vector_dot_product_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_[0].size(); i++) {
    res += input_[0][i] * input_[1][i];
  }

  return true;
}

bool rezantseva_a_vector_dot_product_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

int rezantseva_a_vector_dot_product_seq::vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
  long long result = 0;
  for (size_t i = 0; i < v1.size(); i++) result += v1[i] * v2[i];
  return result;
}