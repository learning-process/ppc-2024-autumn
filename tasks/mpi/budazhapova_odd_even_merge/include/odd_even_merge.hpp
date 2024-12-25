#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {

class MergeSequential : public ppc::core::Task {
 public:
  explicit MergeSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> res;
  std::vector<int> local_res;
  int n_el = 0;
};
class MergeParallel : public ppc::core::Task {
 public:
  explicit MergeParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> res;
  std::vector<int> local_res;
  int n_el = 0;

  boost::mpi::communicator world;
};
}  // namespace budazhapova_betcher_odd_even_merge_mpi