#include "seq/sharamygina_i_most_different_neighbor_elements/include/ops_seq.hpp"

#include <functional>
#include <random>

bool sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());

  res = abs(input_[0] - input_[1]);

  return true;
}

bool sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::run() {
  internal_order_test();

  for (size_t i = 2; i < input_.size(); ++i) {
    if (res < abs(input_[i] - input_[i - 1])) res = abs(input_[i] - input_[i - 1]);
  }

  return true;
}

bool sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
