#pragma once

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm {

class dense_matrix_multiplication_block_scheme_fox_algorithm_seq : public ppc::core::Task {
 public:
  explicit dense_matrix_multiplication_block_scheme_fox_algorithm_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  size_t column_A, row_A;
  std::vector<double> B;
  size_t column_B, row_B;
  std::vector<double> C;
};

}  // namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm