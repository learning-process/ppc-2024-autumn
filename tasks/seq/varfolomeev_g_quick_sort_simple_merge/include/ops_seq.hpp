// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_quick_sort_simple_merge_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::vector<int> quickSort(const std::vector<int>& arr) {
    std::vector<int> sortedArr = arr;
    quickSortRecursive(sortedArr, 0, sortedArr.size() - 1);
    return sortedArr;
  }

 private:
  static void quickSortRecursive(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int p = arr[(left + right) / 2];
    int i = left;
    int j = right;
    while (i <= j) {
      while (arr[i] < p) i++;
      while (arr[j] > p) j--;
      if (i <= j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        i++;
        j--;
      }
    }
    quickSortRecursive(arr, left, j);
    quickSortRecursive(arr, i, right);
  }

  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace varfolomeev_g_quick_sort_simple_merge_seq