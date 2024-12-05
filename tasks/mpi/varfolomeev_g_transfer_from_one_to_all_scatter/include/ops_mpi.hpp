// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_transfer_from_one_to_all_scatter_mpi {

std::vector<int> getRandomVector(int sz, int a, int b);

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
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_values, local_input_values;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};

class MyScatterTestMPITaskParallel : public ppc::core::Task {
 public:
  explicit MyScatterTestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>
  static void myScatter(const boost::mpi::communicator& world, const std::vector<T>& input_values, T* output_values,
                        int root) {
    int rank = world.rank();
    int world_size = world.size();
    int parent = (rank - 1) / 2;
    int left = (((rank * 2) + (1)) < (int)world_size) ? ((rank * 2) + (1)) : -1;
    int right = (((rank * 2) + (2)) < (int)world_size) ? ((rank * 2) + (2)) : -1;
    // int curr_floor = (int)std::floor(log(rank + 1) / std::numbers::ln2);
    // int max_floor = (int)std::floor(log(world_size) / std::numbers::ln2);
    int max_floor = static_cast<int>(std::floor(std::log2(world.size())));
    int curr_floor = static_cast<int>(std::floor(std::log2(world.rank() + 1)));

    if (max_floor - curr_floor != 0) {  // If we're already on max_level, we don't need to send data
      int h = world_size - (std::pow(2, curr_floor + 1) - 1) - 1;
      std::cout << "rank = " << rank << " h = " << h << std::endl;
    }

    std::cout << "rank = " << rank << " parent = " << parent << " left = " << left << " right = " << right
              << " curr_floor = " << curr_floor << " max_floor = " << max_floor << std::endl;
  }

 private:
  std::vector<int> input_values, local_input_values;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};
}  // namespace varfolomeev_g_transfer_from_one_to_all_scatter_mpi