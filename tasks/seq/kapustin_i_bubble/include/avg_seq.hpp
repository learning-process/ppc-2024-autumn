#pragma once

#include <algorithm>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_bubble_sort_seq {

class BubbleSortSequential : public ppc::core::Task {
 public:
  explicit BubbleSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;

  bool pre_processing() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace kapustin_i_bubble_sort_seq