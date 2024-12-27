#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_min_of_vector_elements_mpi {

class MinOfVectorElementTaskSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorElementTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};

class MinOfVectorElementTaskParallel : public ppc::core::Task {
 public:
  explicit MinOfVectorElementTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  unsigned int delta = 0;
  std::vector<int> input_, local_input_;
  int res{};
  boost::mpi::communicator world;
};

}  // namespace komshina_d_min_of_vector_elements_mpi