#pragma once

#include <cstdlib>
#include <cstring>
#include <utility>
#include <algorithm>
#include <cmath>

#include "core/task/include/task.hpp"

namespace kovalev_k_bubble_sort_oddeven_transposition_seq {

template <class T>
class BubbleSortOddEvenTransposition : public ppc::core::Task {
 private:
  std::vector<T> v;
  size_t n;

 public:
  explicit BubbleSortOddEvenTransposition(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), n(taskData_->inputs_count[0]) {}
  bool bubble_sort(T*, size_t);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_bubble_sort_oddeven_transposition_seq