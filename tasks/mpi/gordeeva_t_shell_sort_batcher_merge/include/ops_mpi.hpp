#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gordeeva_t_shell_sort_batcher_merge_mpi {

void shellSort(std::vector<int>& arr);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void batcher_merge(size_t rank1, size_t rank2, std::vector<int>& local_input_local);

 private:
  std::vector<int> input_, local_input_;
  std::vector<int> res_;
  size_t sz_mpi = 0;
  boost::mpi::communicator world;
};

}  // namespace gordeeva_t_shell_sort_batcher_merge_mpi