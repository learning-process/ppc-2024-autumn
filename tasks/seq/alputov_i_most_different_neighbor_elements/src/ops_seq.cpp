#include "seq/alputov_i_most_different_neighbor_elements/include/ops_seq.hpp"

#include <functional>
#include <random>

bool alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::pre_processing() {
  internal_order_test();

  auto input = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], input.begin());

  input_ = std::vector<std::pair<int, int>>(input.size() - 1);

  for (size_t i = 1; i < input.size(); ++i) {
    input_[i - 1] = {std::abs(input[i] - input[i - 1]), std::min(input[i], input[i - 1])};
  }

  res = input_[0];

  return true;
}

bool alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::run() {
  internal_order_test();

  for (size_t i = 1; i < input_.size(); ++i) {
    if (res.first < input_[i].first) res = input_[i];
  }

  return true;
}

bool alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq::post_processing() {
  internal_order_test();

  reinterpret_cast<std::pair<int, int>*>(taskData->outputs[0])[0] = {res.second, res.second + res.first};
  return true;
}
