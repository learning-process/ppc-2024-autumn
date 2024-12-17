// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_bellman_ford_mpi {

class BellmanFordMPITaskParallel : public ppc::core::Task {
 public:
  explicit BellmanFordMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> graph_;
  std::vector<int> local_graph_;
  std::vector<int> dist_;
  boost::mpi::communicator world;
};

}  // namespace zinoviev_a_bellman_ford_mpi