#include "seq/mironov_a_max_of_vector_elements/include/ops_seq.hpp"
#include <thread>

using namespace std::chrono_literals;

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < input_.size(); ++i) {
    input_[i] = it[i];
  }
  res = input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::run() {
  internal_order_test();
  
  for (size_t i = 1; i < input_.size(); ++i) {
    if (res < input_[i]) res = input_[i];
  }
  return true;
}

bool mironov_a_max_of_vector_elements_seq::MaxVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
