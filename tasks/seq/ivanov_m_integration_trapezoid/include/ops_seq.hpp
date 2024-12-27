// Copyright 2024 Ivanov Mike
#pragma once

#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivanov_m_integration_trapezoid_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  void add_function(const ::std::function<double(double)>& f);

 private:
  double a_{}, b_{}, result_{};
  int n_{};
  std::function<double(double)> f_;
};

}  // namespace ivanov_m_integration_trapezoid_seq