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

  static void quickSort(std::vector<int>& vec) { quickSortRecursive(vec, 0, vec.size() - 1); }

 private:
  // main realization
  static void quickSortRecursive(std::vector<int>& vec, int left, int right) {
    if (left >= right) return;
    int p = vec[(left + right) / 2];
    int i = left;
    int j = right;
    while (i <= j) {
      while (vec[i] < p) i++;
      while (vec[j] > p) j--;
      if (i <= j) {
        std::swap(vec[i], vec[j]);
        i++;
        j--;
      }
    }
    quickSortRecursive(vec, left, j);
    quickSortRecursive(vec, i, right);
  }

  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace varfolomeev_g_quick_sort_simple_merge_seq