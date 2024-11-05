#include "seq/gromov_a_sum_of_vector_elements/include/ops_seq.hpp"

bool gromov_a_sum_of_vector_elements_seq::SumOfVector::pre_processing() {
  internal_order_test();
  // Init value for input and output
  res = 0;
  return true;
}

bool gromov_a_sum_of_vector_elements_seq::SumOfVector::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool gromov_a_sum_of_vector_elements_seq::SumOfVector::run() {
  internal_order_test();
  int* inputPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  int count = taskData->inputs_count[0];
  for (int i = 0; i < count; ++i) {
    res += inputPtr[i];
  }
  return true;
}

bool gromov_a_sum_of_vector_elements_seq::SumOfVector::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
