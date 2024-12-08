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

namespace shkurinskaya_e_gauss_jordan_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n = 0;
  std::vector<double> matrix;
  std::vector<double> solution;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n = 0;
  std::vector<double> matrix;
  std::vector<double> local_matrix;
  std::vector<double> solution;
  boost::mpi::communicator world;
  std::vector<double> diag_elements;
  std::vector<double> localMatrix;
  std::vector<double> header;
  std::vector<int> sendCounts;
  std::vector<int> displacements;
};

}  // namespace shkurinskaya_e_gauss_jordan_mpi
