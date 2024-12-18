// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_radix_sort_batcher_merge_seq {

void radixSortDouble(std::vector<double>& arr, int start, int end);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data_;
};

}  // namespace kondratev_ya_radix_sort_batcher_merge_seq