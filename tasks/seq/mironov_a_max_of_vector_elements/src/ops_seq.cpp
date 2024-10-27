#include "seq/mironov_a_max_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_.begin());
  result_ = input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::run() {
  internal_order_test();
  result_ = input_[0];
  for (size_t it = 1; it < input_.size(); ++it) {
    if (result_ < input_[it]) result_ = input_[it];
  }
  return true;
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  return true;
}
