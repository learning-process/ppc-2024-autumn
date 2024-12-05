// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_simple_iteration_method_seq {

class SlaeIterationTask : public ppc::core::Task {
 public:
  explicit SlaeIterationTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_size_;
  double tolerance_;
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  std::vector<std::vector<double>> B_;
  std::vector<double> g_;

  // Helper functions
  bool isDiagonallyDominant();
  bool transformSystem();
};

}  // namespace lysov_i_simple_iteration_method_seq