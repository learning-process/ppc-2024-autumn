// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_sum_of_rows {

class SumOfRowsTaskSequential : public ppc::core::Task {
 public:
  explicit SumOfRowsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> row_sums_;
};

}  // namespace borisov_s_sum_of_rows