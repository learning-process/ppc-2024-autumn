#include "seq/durynichev_d_most_different_neighbor_elements/include/ops_seq.hpp"

bool durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
}

bool durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  auto *input_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input.assign(input_ptr, input_ptr + input_size);
  result.resize(2);
  return true;
}

bool durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  result[0] = input[0];
  result[1] = input[1];
  int maxDiff = std::abs(input[0] - input[1]);

  for (size_t i = 1; i < input.size(); ++i) {
    int diff = std::abs(input[i] - input[i - 1]);
    if (diff > maxDiff) {
      maxDiff = diff;
      result[0] = std::min(input[i], input[i - 1]);
      result[1] = std::max(input[i], input[i - 1]);
    }
  }
  return true;
}

bool durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy_n(result.begin(), 2, reinterpret_cast<int *>(taskData->outputs[0]));
  return true;
}