#pragma once

#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_matrix_column_max_value_seq {

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

}  // namespace nasedkin_e_matrix_column_max_value_seq