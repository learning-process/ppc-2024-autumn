// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_radix_sort_batcher_merge_mpi {

void radixSortDouble(std::vector<double>& arr, int32_t start, int32_t end);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::vector<double> distribute_data(const std::vector<double>& input, int32_t size, int32_t rank);
  void exchange_and_merge(int32_t rank1, int32_t size1, int32_t rank2, int32_t size2);
  static void merge(const std::vector<double>& first, const std::vector<double>& second, std::vector<double>& result);

 private:
  std::vector<double> input_, local_input_;
  std::vector<double> res_;
  int32_t size_;
  boost::mpi::communicator world;
};

}  // namespace kondratev_ya_radix_sort_batcher_merge_mpi