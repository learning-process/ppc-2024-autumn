// Copyright 2024 Nesterov Alexander
#include "seq/khovansky_d_max_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <thread>

using namespace std::chrono_literals;

int khovansky_d_max_of_vector_elements_seq::VectorMax(std::vector<int, std::allocator<int>> r) {
  int max = r[0];

  for (size_t i = 1; i < r.size(); i++) {
    if (r[i] > max) {
      max = r[i];
    }
  }

  return max;
}

bool khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());

  return true;
}

bool khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq::validation() {
  internal_order_test();
  // Check count elements of output
  if (taskData->inputs_count[0] == 0) {
    return taskData->outputs_count[0] == 0;
  }
  return taskData->outputs_count[0] == 1;
}

bool khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq::run() {
  internal_order_test();
  res = khovansky_d_max_of_vector_elements_seq::VectorMax(input_);
  return true;
}

bool khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
