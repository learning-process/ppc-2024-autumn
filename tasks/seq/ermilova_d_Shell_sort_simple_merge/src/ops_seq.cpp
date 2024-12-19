// Copyright 2024 Nesterov Alexander
#include "seq/ermilova_d_Shell_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

std::vector<int> ermilova_d_Shell_sort_simple_merge_seq::ShellSort(std::vector<int>& vec) {
  size_t n = vec.size();

  std::vector<size_t> Sedgwick_gaps;

  size_t a = 0;
  while (true) {
    size_t gap;
    if (a % 2 == 0) {
      gap = 9 * static_cast<size_t>(pow(2, a)) - 9 * static_cast<size_t>(pow(2, a / 2)) + 1;
    } else {
      gap = 8 * static_cast<size_t>(pow(2, a)) - 6 * static_cast<size_t>(pow(2, (a + 1) / 2)) + 1;
    }

    if (gap >= n) break;
    Sedgwick_gaps.push_back(gap);
    a++;
  }

  for (int k = Sedgwick_gaps.size() - 1; k >= 0; --k) {
    size_t gap = Sedgwick_gaps[k];

    for (size_t i = gap; i < n; i++) {
      int temp = vec[i];
      size_t j;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
  return vec;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
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
  res = ShellSort(input_);

  return true;
}

bool ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), data);
  return true;
}
