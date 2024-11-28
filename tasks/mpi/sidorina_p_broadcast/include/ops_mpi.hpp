#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"

namespace sidorina_p_broadcast_mpi {

class Broadcast : public ppc::core::Task {
 public:
  explicit Broadcast(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  std::function<void(const boost::mpi::communicator&, int*, int, int)> broadcast_fn;

 private:
  std::vector<int> arr;
  std::vector<int> term;
  std::vector<int> res;
  int del = 0;
  int sz = 0;
  boost::mpi::communicator world;
};
}  // namespace sidorina_p_broadcast_mpi