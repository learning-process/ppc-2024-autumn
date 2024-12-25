#include "seq/tyurin_m_shell_sort_batcher_merge/include/ops_seq.hpp"

namespace tyurin_m_shell_sort_batcher_merge_seq {

bool ShellSortBatcherMerge::validation() {
  internal_order_test();

  int val_n = taskData->inputs_count[0];

  return val_n > 0 && val_n % 2 == 0;
}

bool ShellSortBatcherMerge::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_vector.assign(input_ptr, input_ptr + n);
  result = input_vector;

  return true;
}

bool ShellSortBatcherMerge::run() {
  internal_order_test();

  shellSort(result.begin(), result.end());

  return true;
}

bool ShellSortBatcherMerge::post_processing() {
  internal_order_test();

  auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), output_ptr);

  return true;
}

}  // namespace tyurin_m_shell_sort_batcher_merge_seq
