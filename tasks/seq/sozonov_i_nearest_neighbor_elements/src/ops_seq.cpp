#include "seq/sozonov_i_nearest_neighbor_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0;
  return true;
}

bool sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 2;
}

bool sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  int min = INT_MAX;
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    if (abs(input_[i + 1] - input_[i]) < min) {
      min = abs(input_[i + 1] - input_[i]);
      res = i;
    }
  }
  return true;
}

bool sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = input_[res];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = input_[res + 1];
  return true;
}