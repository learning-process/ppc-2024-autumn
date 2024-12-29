// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vedernikova_k_star_topology_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::vector<int> make_random_vector(size_t sz);

 private:
  std::vector<int> data;
  std::vector<int> output;
  int dest;
  boost::mpi::communicator world;
};

}  // namespace vedernikova_k_star_topology_mpi