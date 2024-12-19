#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::string ops;
  int* input_A;
  int* input_B;
  int columns_A;
  int rows_A;
  int columns_B;
  int rows_B;

  std::vector<int> C;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int column_A;
  int row_A;
  int column_B;
  int row_B;

 private:
  std::vector<int> input_, local_input_;
  std::string ops;
  boost::mpi::communicator world;
  int* input_A;
  int* input_B;
  int columns_A;
  int rows_A;
  int columns_B;
  int rows_B;

  int* C;
};

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi