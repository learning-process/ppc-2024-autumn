#pragma once

#include <boost/mpi.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_bubble_sort_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  boost::mpi::communicator world;
  int total_size_;
};

}  // namespace naumov_b_bubble_sort_mpi
