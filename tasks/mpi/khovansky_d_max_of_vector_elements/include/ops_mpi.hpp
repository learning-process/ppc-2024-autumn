// Copyright 2024 Khovansky Dmitry
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

namespace khovansky_d_max_of_vector_elements_mpi {

int VectorMax(std::vector<int, std::allocator<int>> r);
std::vector<int> GetRandomVector(int sz, int left, int right);

class MaxOfVectorMPISequential : public ppc::core::Task {
 public:
  explicit MaxOfVectorMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};

class MaxOfVectorMPIParallel : public ppc::core::Task {
 public:
  explicit MaxOfVectorMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res_{};
  boost::mpi::communicator world_;
};

}  // namespace khovansky_d_max_of_vector_elements_mpi
