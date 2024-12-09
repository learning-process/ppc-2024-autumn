// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>
#include <math.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rysev_m_gypercube {

class GyperCube : public ppc::core::Task {
 public:
  explicit GyperCube(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int Next(int c_node, int _target);

 private:
  boost::mpi::communicator world;
  int data;
  int sender;
  int target;
  std::vector<int> path;
  bool done;
};

}  // namespace rysev_m_gypercube