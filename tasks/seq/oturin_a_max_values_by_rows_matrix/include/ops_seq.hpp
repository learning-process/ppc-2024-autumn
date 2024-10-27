#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_max_values_by_rows_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t n = 0;
  size_t m = 0;
  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace oturin_a_max_values_by_rows_matrix_seq