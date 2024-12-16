// Copyright 2024 Nesterov Alexander
#include "seq/koshkin_m_scalar_product_of_vectors/include/ops_seq.hpp"

bool koshkin_m_scalar_product_of_vectors::VectorDotProduct::pre_processing() {
  internal_order_test();
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

bool koshkin_m_scalar_product_of_vectors::VectorDotProduct::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 2 && taskData->inputs.size() == taskData->inputs_count.size() &&
          taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs.size() == 1 &&
          taskData->outputs.size() == taskData->outputs_count.size() && taskData->outputs_count[0] == 1);
}

bool koshkin_m_scalar_product_of_vectors::VectorDotProduct::run() {
  internal_order_test();
  for (size_t i = 0; i < input_[0].size(); i++) {
    res += input_[0][i] * input_[1][i];
  }
  return true;
}

bool koshkin_m_scalar_product_of_vectors::VectorDotProduct::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}