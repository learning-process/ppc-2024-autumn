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

namespace shishkarev_a_sum_of_vector_elements_mpi {

std::vector<int> getRandomVector(int vector_size);

class MPIVectorSumSequential : public ppc::core::Task {
 public:
  explicit MPIVectorSumSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  int result{};
  std::string operation;
};

class MPIVectorSumParallel : public ppc::core::Task {
 public:
  explicit MPIVectorSumParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector, local_vector;
  int result{}, local_sum;
  std::string operation;
  boost::mpi::communicator world;
};

}  // namespace shishkarev_a_sum_of_vector_elements_mpi
