// Copyright 2023 Nesterov Alexander
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_max_of_matrix_elements_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  int res{};
};

}  // namespace opolin_d_max_of_matrix_elements_seq