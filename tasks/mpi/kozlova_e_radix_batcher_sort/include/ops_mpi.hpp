// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kozlova_e_radix_batcher_sort_mpi {

class RadixBatcherSortSequential : public ppc::core::Task {
 public:
  explicit RadixBatcherSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_size{};
  std::vector<double> data;

  static void radixSort(std::vector<double>& a);
};

class RadixBatcherSortMPI : public ppc::core::Task {
 public:
  explicit RadixBatcherSortMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_;
  int input_size{};
  boost::mpi::communicator world;

  static void radixSort(std::vector<double>& a);
  void RadixSortWithOddEvenMerge(std::vector<double>& a);
};

}  // namespace kozlova_e_radix_batcher_sort_mpi