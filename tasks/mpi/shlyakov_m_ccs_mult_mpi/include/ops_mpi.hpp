// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace shlyakov_m_ccs_mult_mpi {

struct SparseMatrix {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;
};

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  SparseMatrix check_result() { return result_; };

 private:
  SparseMatrix A_;
  int rows_a;
  int cols_a;
  SparseMatrix B_;
  int rows_b;
  int cols_b;
  SparseMatrix result_;

  boost::mpi::communicator world;
  boost::mpi::environment env;
};

}  // namespace shlyakov_m_ccs_mult_mpi