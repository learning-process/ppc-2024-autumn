#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace matthew_fyodorov_reduce_custom_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  template <typename T>
  void my_reduce(boost::mpi::communicator& world, int& local_sum, int& global_sum) {
    int rank = world.rank();
    int size = world.size();

    global_sum = local_sum;

    for (int step = 1; step < size; step *= 2) {
      if (rank % (2 * step) == 0) {
        int partner = rank + step;
        if (partner < size) {
          int received_sum;
          world.recv(partner, 0, received_sum);
          global_sum += received_sum;
        }
      } else if (rank % (2 * step) == step) {
        int partner = rank - step;
        world.send(partner, 0, global_sum);
      }
    }
  }

 private:
  std::vector<int> input_, local_input_;
  int res{};
  boost::mpi::communicator world;
};
}  // namespace matthew_fyodorov_reduce_custom_mpi
