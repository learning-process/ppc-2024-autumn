// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_Fox_algorithm_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C;
  std::size_t N;
  std::size_t block_size;
};

}  // namespace lysov_i_matrix_multiplication_Fox_algorithm_seq