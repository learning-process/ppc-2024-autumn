#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_broadcast_mpi {

template <typename T>
void broadcast(const boost::mpi::communicator &comm, T &value, int root) {
  int rank = comm.rank();
  int size = comm.size();

  auto to_tree_rank = [&](int global_rank) { return (global_rank - root + size) % size; };

  auto to_global_rank = [&](int tree_rank) { return (root + tree_rank) % size; };

  auto parent = [&](int tree_r) {
    if (tree_r == 0) return -1;
    return (tree_r - 1) / 2;
  };

  auto left_child = [&](int tree_r) {
    int c = 2 * tree_r + 1;
    return (c < size) ? c : -1;
  };

  auto right_child = [&](int tree_r) {
    int c = 2 * tree_r + 2;
    return (c < size) ? c : -1;
  };

  int tree_rank = to_tree_rank(rank);
  int p = parent(tree_rank);

  if (p != -1) {
    int global_parent = to_global_rank(p);
    comm.recv(global_parent, 0, value);
  }

  int lc = left_child(tree_rank);
  if (lc != -1) {
    int global_lc = to_global_rank(lc);
    comm.send(global_lc, 0, value);
  }

  int rc = right_child(tree_rank);
  if (rc != -1) {
    int global_rc = to_global_rank(rc);
    comm.send(global_rc, 0, value);
  }
}

class MyBroadcastMPI : public ppc::core::Task {
 public:
  explicit MyBroadcastMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  int global_max;
  int source_worker;
  boost::mpi::communicator world;
};

}  // namespace burykin_m_broadcast_mpi
