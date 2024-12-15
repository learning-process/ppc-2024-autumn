#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_linear_topology {

template <typename T>
class LinearTopology : public ppc::core::Task {
 public:
  explicit LinearTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<T> data_;
  bool result_ = false;
};

}  // namespace agafeev_s_linear_topology