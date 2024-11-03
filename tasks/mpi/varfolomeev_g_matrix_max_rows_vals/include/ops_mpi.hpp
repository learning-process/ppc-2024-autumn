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

namespace varfolomeev_g_matrix_max_rows_vals_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_ /*ver link of chars*/, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override; // task data to vec of vecs or general vec
  bool validation() override; // input data check (n > 0, mem of enter and exit)
  bool run() override; // 
  bool post_processing() override; // vec to taskdata

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace varfolomeev_g_matrix_max_rows_vals_mpi