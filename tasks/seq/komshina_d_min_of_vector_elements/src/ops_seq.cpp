#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
    input_[i] = ptr[i];
  }
  res = input_[0];
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] == 1) ||
         (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0);
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::run() {
  internal_order_test();
  res = input_[0];
  for (size_t ptr = 1; ptr < input_.size(); ++ptr) {
    if (res > input_[ptr]) {
      res = input_[ptr];
    }
  }
  return true;
}

bool komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}