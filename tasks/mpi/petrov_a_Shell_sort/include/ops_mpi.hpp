#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_a_Shell_sort_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_;
};

}  // namespace petrov_a_Shell_sort_mpi
