#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/status.hpp>
#include <random>

#include "core/task/include/task.hpp"

namespace lopatin_i_quick_batcher_mergesort_mpi {

void quicksort(std::vector<int>& arr, int low, int high);
int partition(std::vector<int>& arr, int low, int high);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputArray_;
  std::vector<int> resultArray_;

  int sizeArray;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputArray_;
  std::vector<int> resultArray_;

  std::vector<int> localArray;

  int sizeArray;

  boost::mpi::communicator world;
};

}  // namespace lopatin_i_quick_batcher_mergesort_mpi