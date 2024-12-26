#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace durynichev_d_custom_allreduce_mpi {

template <typename T, typename BinaryOperation>
T custom_all_reduce(boost::mpi::communicator& world, const T& local_value, BinaryOperation op) {
  int rank = world.rank();
  int size = world.size();
  T global_result = local_value;

  for (int step = 1; step < size; step *= 2) {
    int partner = rank ^ step;

    if (partner < size) {
      T received_value;
      if (rank < partner) {
        world.send(partner, 0, global_result);
        world.recv(partner, 0, received_value);
      } else {
        world.recv(partner, 0, received_value);
        world.send(partner, 0, global_result);
      }
      global_result = op(global_result, received_value);
    }
  }

  int tree_rank = (rank + size) % size;

  int p = (tree_rank == 0) ? -1 : (tree_rank - 1) / 2;
  if (p != -1) {
    int global_parent = (p + size) % size;
    world.recv(global_parent, 0, global_result);
  }

  int lc = 2 * tree_rank + 1;
  if (lc < size) {
    int global_lc = (lc + size) % size;
    world.send(global_lc, 0, global_result);
  }

  int rc = 2 * tree_rank + 2;
  if (rc < size) {
    int global_rc = (rc + size) % size;
    world.send(global_rc, 0, global_result);
  }

  return global_result;
}

template <typename T, typename BinaryOperation>
void custom_all_reduce(boost::mpi::communicator& world, const T& local_value, T& global_result, BinaryOperation op) {
  global_result = custom_all_reduce(world, local_value, op);
}

class MyAllreduceMPI : public ppc::core::Task {
 public:
  explicit MyAllreduceMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data;
  int global_sum = 0;
  boost::mpi::communicator world;
};

}  // namespace durynichev_d_custom_allreduce_mpi
