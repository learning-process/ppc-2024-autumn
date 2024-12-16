#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_n_sum_values_by_columns_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res;
  int rows;
  int columns;
};
}  // namespace koshkin_n_sum_values_by_columns_matrix_seq