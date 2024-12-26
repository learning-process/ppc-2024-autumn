// Copyright 2024 Sdobnov Vladimir
#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace Sdobnov_V_mergesort_Betcher_par {

std::vector<int> generate_random_vector(int size, int lower_bound = 0, int upper_bound = 50);
int partition(std::vector<int>& vec, int low, int high);
void quickSortIterative(std::vector<int>& vec, int low, int high);

class MergesortBetcherPar : public ppc::core::Task {
 public:
  explicit MergesortBetcherPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int size_;
  std::vector<int> local_vec_;

  boost::mpi::communicator world;
};
}  // namespace Sdobnov_V_mergesort_Betcher_par