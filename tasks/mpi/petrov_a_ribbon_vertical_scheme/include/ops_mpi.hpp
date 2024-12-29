// Copyright 2024 Nesterov Alexander
#pragma once

#include <mpi.h>

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_a_ribbon_vertical_scheme_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> local_matrix;
  std::vector<int> local_vector;
  std::vector<int> local_result;
};

}  // namespace petrov_a_ribbon_vertical_scheme_mpi
