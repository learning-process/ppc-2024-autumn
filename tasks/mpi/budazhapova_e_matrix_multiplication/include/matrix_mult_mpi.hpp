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

namespace budazhapova_e_matrix_mult_mpi {

class MatrixMultSequential : public ppc::core::Task {
 public:
  explicit MatrixMultSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{};
  int columns{};

  std::vector<int> A;
  std::vector<int> b;
  std::vector<int> res;
};

class MatrixMultParallel : public ppc::core::Task {
 public:
  explicit MatrixMultParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{};
  int columns{};

  std::vector<int> A;
  std::vector<int> b;
  std::vector<int> res;

  std::vector<int> local_res;
  std::vector<int> local_A;

  boost::mpi::communicator world;
};

}  // namespace budazhapova_e_matrix_mult_mpi