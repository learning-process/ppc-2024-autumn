// Copyright 2024 Nesterov Alexander
#include "seq/muhina_m_shell_sort/include/ops_seq.hpp"

#include <random>
#include <thread>

std::vector<int> muhina_m_shell_sort_seq::shellSort(const std::vector<int>& vect) {
  std::vector<int> sortedVec = vect;
  int n = sortedVec.size();
  int gap;
  for (gap = 1; gap < n / 3; gap = gap * 3 + 1);
  for (; gap > 0; gap = (gap - 1) / 3) {
    for (int i = gap; i < n; i++) {
      int temp = sortedVec[i];
      int j;
      for (j = i; j >= gap && sortedVec[j - gap] > temp; j -= gap) {
        sortedVec[j] = sortedVec[j - gap];
      }
      sortedVec[j] = temp;
    }
  }
  return sortedVec;
}

bool muhina_m_shell_sort_seq::ShellSortSequential::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tempPtr[i];
  }

  return true;
}

bool muhina_m_shell_sort_seq::ShellSortSequential::validation() {
  internal_order_test();
  int sizeVec = taskData->inputs_count[0];
  int sizeResultVec = taskData->outputs_count[0];

  return (sizeVec > 0 && sizeVec == sizeResultVec);
}

bool muhina_m_shell_sort_seq::ShellSortSequential::run() {
  internal_order_test();
  res_ = shellSort(input_);
  return true;
}

bool muhina_m_shell_sort_seq::ShellSortSequential::post_processing() {
  internal_order_test();
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_data);
  return true;
}
