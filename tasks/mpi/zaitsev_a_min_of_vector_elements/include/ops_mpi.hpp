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

namespace zaitsev_a_min_of_vector_elements_mpi {

std::vector<int> getRandomVector(int sz, int minRangeValue, int maxRangeValue);

class MinOfVectorElementsSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorElementsSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input{};
  int res{};
};

class MinOfVectorElementsParallel : public ppc::core::Task {
 public:
  explicit MinOfVectorElementsParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  boost::mpi::communicator world;
};

}  // namespace zaitsev_a_min_of_vector_elements_mpi