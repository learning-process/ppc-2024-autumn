#include "seq/polikanov_v_max_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace std::chrono_literals;

bool polikanov_v_max_of_vector_elements::TestTaskSequential::pre_processing() {
  internal_order_test();
  int count = static_cast<int>(taskData->inputs_count[0]);
  int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(count);
  std::copy(input, input + count, input_.begin());
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0) || taskData->outputs_count[0] == 1;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::run() {
  internal_order_test();
  int count = input_.size();
  for (int i = 0; i < count; i++) {
    res = std::max(res, input_[i]);
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
