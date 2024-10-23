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

namespace mironov_a_max_of_vector_elements_mpi {

class MaxVectorSequential : public ppc::core::Task {
 public:
  explicit MaxVectorSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};

class MaxVectorMPI : public ppc::core::Task {
 public:
  explicit MaxVectorMPI(std::shared_ptr<ppc::core::TaskData> taskData_): Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  boost::mpi::communicator world;
};

}  // namespace mironov_a_max_of_vector_elements_mpi