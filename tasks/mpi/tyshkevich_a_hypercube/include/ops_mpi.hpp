#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace tyshkevich_a_hypercube_mpi {

inline int getNextNode(int current, int target, int n) {
  int diff = current ^ target;
  for (int i = 0; i < n; ++i) {
    if ((diff & (1 << i)) != 0) {
      return current ^ (1 << i);
    }
  }
  return current;
}

class HypercubeParallelMPI : public ppc::core::Task {
 public:
  explicit HypercubeParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> shortest_route;
  int route_iters;
  int dimension;
  int sender_id;
  int target_id;
  std::vector<int> message;
  std::vector<int> result;
  boost::mpi::communicator world;
};

}  // namespace tyshkevich_a_hypercube_mpi
