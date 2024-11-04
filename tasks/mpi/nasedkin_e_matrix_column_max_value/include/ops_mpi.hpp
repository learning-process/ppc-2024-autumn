#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "core/task/include/task.hpp"

namespace nasedkin_e_matrix_column_max_value_mpi {

std::vector<int> getRandomMatrix(int rows, int cols);

class MatrixColumnMaxMPI : public ppc::core::Task {
 public:
  explicit MatrixColumnMaxMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> local_input_;
  std::vector<int> res_;
  boost::mpi::communicator world;
};

}  // namespace nasedkin_e_matrix_column_max_value_mpi