#pragma once

#include <boost/mpi/collectives.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace rams_s_radix_sort_with_simple_merge_for_doubles_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input;
  std::vector<double> result;
  boost::mpi::communicator world;
};

}  // namespace rams_s_radix_sort_with_simple_merge_for_doubles_mpi
