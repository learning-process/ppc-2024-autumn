#include "seq/chernykh_a_num_of_alternations_signs/include/ops_seq.hpp"

bool chernykh_a_num_of_alternations_signs_seq::Task::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 1;
}

bool chernykh_a_num_of_alternations_signs_seq::Task::pre_processing() {
  internal_order_test();
  auto *input_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input = std::vector<int>(input_ptr, input_ptr + input_size);
  result = 0;
  return true;
}

bool chernykh_a_num_of_alternations_signs_seq::Task::run() {
  internal_order_test();
  auto input_size = input.size();
  for (size_t i = 0; i < input_size - 1; i++) {
    if ((input[i] ^ input[i + 1]) < 0) {
      result++;
    }
  }
  return true;
}

bool chernykh_a_num_of_alternations_signs_seq::Task::post_processing() {
  internal_order_test();
  *reinterpret_cast<int *>(taskData->outputs[0]) = result;
  return true;
}
