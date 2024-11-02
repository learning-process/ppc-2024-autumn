// Copyright 2024 Nesterov Alexander
#include "seq/matyunina_a_average_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool matyunina_a_average_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  res_ = 0;
  return true;
}

bool matyunina_a_average_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool matyunina_a_average_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  res_ = std::accumulate(input_.begin(), input_.end(), 0);
  res_ /= static_cast<int>(input_.size());
  return true;
}

bool matyunina_a_average_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  return true;
}
