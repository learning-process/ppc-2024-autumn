#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_star_topology_mpi {
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
   boost::mpi::communicator world;
   std::vector<int> input_;
   std::vector<int> traversal_order_;
};

}  // namespace bessonov_e_star_topology_mpi