// Filatev Vladislav Sum_of_matrix_elements
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

namespace filatev_v_sum_of_matrix_elements_mpi {

std::vector<std::vector<int>> getRandomMatrix(int size_n, int size_m);

class SumMatrixSeq : public ppc::core::Task {
 public:
  explicit SumMatrixSeq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix;
  long long summ = 0;
  int size_n, size_m;
};

class SumMatrixParallel : public ppc::core::Task {
 public:
  explicit SumMatrixParallel(std::shared_ptr<ppc::core::TaskData> taskData_, boost::mpi::communicator world): Task(std::move(taskData_)), world(world) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix;
  long long summ = 0;
  std::vector<int> local_vector;
  int size_n, size_m;
  boost::mpi::communicator world;
};

}  // namespace filatev_v_sum_of_matrix_elements_mpi