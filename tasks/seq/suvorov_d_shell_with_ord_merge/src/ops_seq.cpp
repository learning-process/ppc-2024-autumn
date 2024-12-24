// Copyright 2024 Nesterov Alexander
#include "seq/suvorov_d_shell_with_ord_merge/include/ops_seq.hpp"

std::vector<int> suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq::shell_sort(const std::vector<int>& vec_to_sort) {
  std::vector<int> result_vec = vec_to_sort;
  size_t n = result_vec.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; ++i) {
      int curr_val = result_vec[i];
      size_t j = i;

      while (j >= gap && result_vec[j - gap] > curr_val) {
        result_vec[j] = result_vec[j - gap];
        j -= gap;
      }
      result_vec[j] = curr_val;
    }
  }

  return result_vec;
}

bool suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq::pre_processing() {
  internal_order_test();

  auto data_size = static_cast<size_t>(taskData->inputs_count[0]);
  int* data_tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  data_to_sort.assign(data_tmp_ptr, data_tmp_ptr + data_size);

  return true;
}

bool suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] > 0 && taskData->inputs_count.size() == 1 &&
         taskData->outputs_count.size() == 1 && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq::run() {
  internal_order_test();

  sorted_data = shell_sort(data_to_sort);

  return true;
}

bool suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq::post_processing() {
  internal_order_test();

  std::copy(sorted_data.begin(), sorted_data.end(), reinterpret_cast<int*>(taskData->outputs[0]));

  return true;
}
