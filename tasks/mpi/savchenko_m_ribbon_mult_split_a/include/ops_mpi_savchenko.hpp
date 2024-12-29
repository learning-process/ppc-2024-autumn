#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace savchenko_m_ribbon_mult_split_a_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  int columns_A, rows_A;
  int columns_B, rows_B;
  std::vector<int> matrix_res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  int size;

  std::vector<int> matrix_res;

  boost::mpi::communicator world;
};

}  // namespace savchenko_m_ribbon_mult_split_a_mpi