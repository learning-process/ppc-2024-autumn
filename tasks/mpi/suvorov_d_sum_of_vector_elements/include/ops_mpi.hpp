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

namespace suvorov_d_sum_of_vector_elements_mpi {

std::vector<int> getRandomVector(int sz);

class Sum_of_vector_elements_seq : public ppc::core::Task {
 public:
  explicit Sum_of_vector_elements_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};

class Sum_of_vector_elements_parallel : public ppc::core::Task {
 public:
  explicit Sum_of_vector_elements_parallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res_{};
  boost::mpi::communicator world_;
};

}  // namespace suvorov_d_sum_of_vector_elements_mpi