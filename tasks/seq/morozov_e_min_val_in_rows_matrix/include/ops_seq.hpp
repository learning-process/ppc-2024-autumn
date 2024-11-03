#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace morozov_e_min_val_in_rows_matrix {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> matrix_;
  std::vector<int> min_val_list_;
};

}  // namespace morozov_e_min_val_in_rows_matrix