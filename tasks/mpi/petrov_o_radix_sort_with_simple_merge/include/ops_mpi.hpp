#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_radix_sort_with_simple_merge_mpi {

class TaskParallel : public ppc::core::Task {
 public:
  explicit TaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int> input_;
  std::vector<int> res;
};

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace petrov_o_radix_sort_with_simple_merge_mpi