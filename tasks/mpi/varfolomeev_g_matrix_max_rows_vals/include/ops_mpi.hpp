#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_matrix_max_rows_vals_mpi {

class MaxInRowsSequential : public ppc::core::Task {
 public:
  explicit MaxInRowsSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size_n, size_m;
  std::vector<int> mtr;
  std::vector<int> res_vec;
};

class MaxInRowsParallel : public ppc::core::Task {
 public:
  explicit MaxInRowsParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size_n, size_m;
  std::vector<int> mtr;      // vector one-row matrix
  std::vector<int> res_vec;  // result vector of maxes
  boost::mpi::communicator world;
};

}  // namespace varfolomeev_g_matrix_max_rows_vals_mpi