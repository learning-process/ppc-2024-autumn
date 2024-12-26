// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"

namespace varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi {

class MyScatterTestMPITaskParallel : public ppc::core::Task {
 public:
  explicit MyScatterTestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>

  static void myScatter(const boost::mpi::communicator& world, const std::vector<T>& input_values,
                        T* local_input_values, int delta, int root) {
    int left = 2 * world.rank() + 1;
    int right = 2 * world.rank() + 2;
    int level = static_cast<int>(std::floor(log(world.rank() + 1) / std::numbers::ln2));
    int neighbours = static_cast<int>(pow(2, level));
    if (world.rank() == root) {
      std::copy(input_values.begin(), input_values.begin() + delta, local_input_values);
      if (left < world.size()) {
        world.send(left, 0, input_values.data() + delta, (world.size() - 1) * delta);
      }
      if (right < world.size()) {
        world.send(right, 0, input_values.data() + delta, (world.size() - 1) * delta);
      }
    } else {
      int minRank = neighbours - 1;
      int receiveSize = (world.size() - minRank) * delta;
      std::vector<T> recv_vec(receiveSize);
      world.recv((world.rank() - 1) / 2, 0, recv_vec.data(), receiveSize);
      std::copy(recv_vec.begin() + (world.rank() - minRank) * delta,
                recv_vec.begin() + (world.rank() - minRank) * delta + delta, local_input_values);
      if (left < world.size()) {
        world.send(left, 0, recv_vec.data() + delta * neighbours, (world.size() - minRank * 2 - 1) * delta);
      }
      if (right < world.size()) {
        world.send(right, 0, recv_vec.data() + delta * neighbours, (world.size() - minRank * 2 - 1) * delta);
      }
    }
  }

 private:
  std::vector<int> input_values, local_input_values;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};
}  // namespace varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi