// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <vector>
#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace lavrentyev_a_line_topology_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data;
  std::vector<int> path;
  boost::mpi::communicator world;
};

}  // namespace lavrentyev_a_line_topology_mpi
