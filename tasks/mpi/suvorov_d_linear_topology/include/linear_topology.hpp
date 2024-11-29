// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace suvorov_d_linear_topology_mpi {

class MPILinearTopology : public ppc::core::Task {
 public:
  explicit MPILinearTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> local_data_, rank_order_, verific_data_;
  boost::mpi::communicator world;
};

}  // namespace suvorov_d_linear_topology_mpi