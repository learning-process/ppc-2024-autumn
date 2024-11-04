#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_min_elements_in_columns_of_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

}  // namespace vavilov_v_min_elements_in_columns_of_matrix_seq
