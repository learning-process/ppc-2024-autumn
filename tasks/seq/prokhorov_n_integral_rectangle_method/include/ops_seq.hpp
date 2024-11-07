// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>
#include <functional>

#include "core/task/include/task.hpp"

namespace prokhorov_n_integral_rectangle_method {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);
 private:
  static double integrate(const std::function<double(double)>& f, double left_, double right_, int n);

  double left_{};
  double right_{};
  int n{};
  double res{};
  std::function<double(double)> func_;
};

}  // namespace prokhorov_n_integral_rectangle_method