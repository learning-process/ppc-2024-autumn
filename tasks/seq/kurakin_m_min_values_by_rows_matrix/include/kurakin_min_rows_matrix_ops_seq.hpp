// Copyright 2023 Nesterov Alexander
#pragma once

#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_min_values_by_rows_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_rows{};
  int size_rows{};
  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace kurakin_m_min_values_by_rows_matrix_seq