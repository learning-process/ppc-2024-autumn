// Copyright 2023 Nesterov Alexander
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

namespace lysov_i_matrix_multiplication_Fox_algorithm_mpi {

std::vector<int> getRandomVector(int sz);
static void extract_submatrix_block(const std::vector<double>& matrix, double* block, int total_columns, int block_size,
                                    int block_row_index, int block_col_index);
static void multiply_matrix_blocks(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
                                   int block_size);
void perform_fox_algorithm_step(boost::mpi::communicator& my_world, int rank, int cnt_work_process, int K,
                                std::vector<double>& local_A, std::vector<double>& local_B,
                                std::vector<double>& local_C);
  class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C;
  int N;
  std::size_t block_size;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int dimension{}, elements{};
  std::vector<double> initialMatrixA;
  std::vector<double> initialMatrixB;
  std::vector<double> resultC;
  boost::mpi::communicator world;
};

}  // namespace lysov_i_matrix_multiplication_Fox_algorithm_mpi