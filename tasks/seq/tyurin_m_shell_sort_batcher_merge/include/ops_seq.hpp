#pragma once

#include <algorithm>
#include <cmath>

#include "core/task/include/task.hpp"

namespace tyurin_m_shell_sort_batcher_merge_seq {

template <typename RandomIt>
void shellSort(RandomIt begin, RandomIt end) {
  auto n = std::distance(begin, end);
  if (n <= 1) return;

  std::vector<int> gaps;
  int k = 0;
  int gap;
  do {
    if (k % 2 == 0) {
      gap = 9 * std::pow(2, k) - 9 * std::pow(2, k / 2) + 1;
    } else {
      gap = 8 * std::pow(2, k) - 6 * std::pow(2, (k + 1) / 2) + 1;
    }
    if (gap < n) gaps.push_back(gap);
    k++;
  } while (gap < n);

  std::sort(gaps.rbegin(), gaps.rend());
  for (int gap2 : gaps) {
    for (auto i = begin + gap2; i != end; ++i) {
      auto temp = *i;
      auto j = i;
      while (j >= begin + gap2 && *(j - gap2) > temp) {
        *j = *(j - gap2);
        j -= gap2;
      }
      *j = temp;
    }
  }
}

class ShellSortBatcherMerge : public ppc::core::Task {
 public:
  explicit ShellSortBatcherMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n;
  std::vector<int> input_vector;
  std::vector<int> result;
};

}  // namespace tyurin_m_shell_sort_batcher_merge_seq