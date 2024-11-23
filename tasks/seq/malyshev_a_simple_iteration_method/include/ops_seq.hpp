// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/malyshev_a_simple_iteration_method/include/matrix.hpp"

namespace malyshev_a_simple_iteration_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> C_;
  std::vector<double> B_;
  std::vector<double> D_;
  std::vector<double> X_;
  std::vector<double> X0_;
  double eps_;
};

}  // namespace malyshev_a_simple_iteration_method_seq