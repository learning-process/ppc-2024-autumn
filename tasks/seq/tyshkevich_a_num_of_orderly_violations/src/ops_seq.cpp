#include "seq/tyshkevich_a_num_of_orderly_violations/include/ops_seq.hpp"

using namespace std::chrono_literals;

bool tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  // Init vectors
  size = taskData->inputs_count[0];

  input_ = std::vector<int>(size);
  int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (int i = 0; i < size; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init values for output
  res = std::vector<int>(1, 0);
  return true;
}

bool tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count of elements in I/O
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 1; i < size; i++) {
    if (input_[i - 1] > input_[i]) res[0]++;
  }
  return true;
}

bool tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res[0];
  return true;
}