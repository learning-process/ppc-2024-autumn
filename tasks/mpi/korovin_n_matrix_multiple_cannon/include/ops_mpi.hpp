#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_matrix_multiple_cannon_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int numRowsA_;
  int numColsA_RowsB_;
  int numColsB_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int numRowsA_;
  int numColsA_RowsB_;
  int numColsB_;
  std::vector<double> A_original_;
  std::vector<double> B_original_;
};

}  // namespace korovin_n_matrix_multiple_cannon_mpi