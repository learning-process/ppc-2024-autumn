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

namespace titov_s_vector_sum_mpi {

std::vector<int> getRandomVector(int sz);

class MPIVectorSumSequential : public ppc::core::Task {
 public:
  explicit MPIVectorSumSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class MPIVectorSumParallel : public ppc::core::Task {
 public:
  explicit MPIVectorSumParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
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

}  // namespace titov_s_vector_sum_mpi