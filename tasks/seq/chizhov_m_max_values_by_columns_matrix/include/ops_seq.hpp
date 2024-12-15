// Copyright 2023 Nesterov Alexander
#pragma once

#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"

namespace chizhov_m_max_values_by_columns_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cols{};
  int rows{};
  std::vector<int> input_;
  std::vector<int> res_;
};

}  // namespace chizhov_m_max_values_by_columns_matrix_seq