#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }
  if (taskData->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::run() {
  internal_order_test();
  res = input_[0];
  for (size_t tmp_ptr = 1; tmp_ptr < input_.size(); ++tmp_ptr) {
    if (res > input_[tmp_ptr]) {
      res = input_[tmp_ptr];
    }
  }
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}