#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_seq {

class MatrixMultiplicationTaskSequential : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_a;
  std::vector<int> matrix_b;
  std::vector<int> matrix_c;

  size_t row_a, col_a;
  size_t row_b, col_b;
};

}  // namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_seq