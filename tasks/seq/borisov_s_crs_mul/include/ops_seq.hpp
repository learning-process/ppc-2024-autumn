#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_crs_mul {

class CrsMatrixMulTask : public ppc::core::Task {
 public:
  explicit CrsMatrixMulTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void initialize(const std::vector<double>& A_values, const std::vector<int>& A_col_index,
                  const std::vector<int>& A_row_ptr, const std::vector<double>& B_values,
                  const std::vector<int>& B_col_index, const std::vector<int>& B_row_ptr, int A_nrows, int A_ncols,
                  int B_nrows, int B_ncols) {
    A_values_ = A_values;
    A_col_index_ = A_col_index;
    A_row_ptr_ = A_row_ptr;

    B_values_ = B_values;
    B_col_index_ = B_col_index;
    B_row_ptr_ = B_row_ptr;

    A_nrows_ = A_nrows;
    A_ncols_ = A_ncols;
    B_nrows_ = B_nrows;
    B_ncols_ = B_ncols;

    C_nrows_ = A_nrows;
    C_ncols_ = B_ncols;
  }

 private:
  std::vector<double> A_values_;
  std::vector<int> A_col_index_;
  std::vector<int> A_row_ptr_;

  std::vector<double> B_values_;
  std::vector<int> B_col_index_;
  std::vector<int> B_row_ptr_;

  std::vector<double> C_values_;
  std::vector<int> C_col_index_;
  std::vector<int> C_row_ptr_;

  int A_nrows_, A_ncols_, A_nnz_;
  int B_nrows_, B_ncols_, B_nnz_;
  int C_nrows_, C_ncols_;
  int C_nnz_;
};

}  // namespace borisov_s_crs_mul