// Copyright 2024 Nesterov Alexander
#include "seq/kolodkin_g_hoar_merge_sort/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

int partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return (i + 1);
}

void quickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

bool kolodkin_g_hoar_merge_sort_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  output_ = std::vector<int>(taskData->inputs_count[0]);
  output_ = input_;
  return true;
}

bool kolodkin_g_hoar_merge_sort_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1;
}

bool kolodkin_g_hoar_merge_sort_seq::TestTaskSequential::run() {
  internal_order_test();
  quickSort(output_, 0, output_.size() - 1);
  return true;
}

bool kolodkin_g_hoar_merge_sort_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  return true;
}
