#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_linear_topology {

std::vector<int> calculating_Route(int a, int b);

class LinearTopology : public ppc::core::Task {
 public:
  explicit LinearTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int> perfect_way_;
  std::vector<int> ranks_vec_;
  bool result_ = false;
  int sender_;
  int receiver_;
};

}  // namespace agafeev_s_linear_topology