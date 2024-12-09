#pragma once

#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_num_of_alternations_signs_mpi {

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input;
  int result{};
};

class ParallelTask : public ppc::core::Task {
 public:
  explicit ParallelTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input, chunk;
  int result{};
  boost::mpi::communicator world;
};

}  // namespace chernykh_a_num_of_alternations_signs_mpi