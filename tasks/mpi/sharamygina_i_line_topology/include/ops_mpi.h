#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_line_topology_mpi {

class line_topology_mpi : public ppc::core::Task {
 public:
  explicit line_topology_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> message;
  boost::mpi::communicator world;
  int sendler;
  int recipient;
  int msize;
};

}  // namespace sharamygina_i_line_topology_mpi