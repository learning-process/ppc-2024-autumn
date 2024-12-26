#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_crs_mul_mpi {

class CrsMatrixMulTaskMPI : public ppc::core::Task {
 public:
  explicit CrsMatrixMulTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

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

  int A_nrows_, A_ncols_;
  int B_nrows_, B_ncols_;
  int C_nrows_, C_ncols_;

  boost::mpi::communicator world;
};

}  // namespace borisov_s_crs_mul_mpi