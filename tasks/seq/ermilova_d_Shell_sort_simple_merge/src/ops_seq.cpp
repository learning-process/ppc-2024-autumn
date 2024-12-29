// Copyright 2024 Nesterov Alexander
#include "seq/ermilova_d_Shell_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

std::vector<int> ermilova_d_Shell_sort_simple_merge_seq::ShellSort(std::vector<int>& vec,
                                                                   const std::function<bool(int, int)>& comp) {
  size_t n = vec.size();
  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = vec[i];
      size_t j;
      for (j = i; j >= gap && comp(vec[j - gap], temp); j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
  return vec;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  is_descending = *reinterpret_cast<bool*>(taskData->inputs[1]);
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(data, data + taskData->inputs_count[0], input_.begin());
  return true;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0] && taskData->inputs_count[0] > 0;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  if (is_descending) {
    res = ShellSort(input_, std::less());
  } else {
    res = ShellSort(input_, std::greater());
  }

  return true;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), data);
  return true;
}
