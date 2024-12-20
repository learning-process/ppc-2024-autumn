#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "core/task/include/task.hpp"

namespace chastov_v_bubble_sort {

template <class T>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), data_size(taskData_->inputs_count[0]) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  bool bubble_sort(T*, size_t);

 private:
  std::vector<T> data;
  size_t data_size;
};

}  // namespace chastov_v_bubble_sort