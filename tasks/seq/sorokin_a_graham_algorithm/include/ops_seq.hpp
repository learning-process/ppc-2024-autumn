// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorokin_a_graham_algorithm_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};

}  // namespace sorokin_a_graham_algorithm_seq