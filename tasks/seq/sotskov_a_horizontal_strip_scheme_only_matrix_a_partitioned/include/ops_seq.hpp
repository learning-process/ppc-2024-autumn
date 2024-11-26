#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* input_A;
  int* input_B;
  int column_A{};
  int row_A{};
  int column_B{};
  int row_B{};
  std::vector<int> C;
};

}  // namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq