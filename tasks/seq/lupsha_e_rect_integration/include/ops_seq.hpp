// Copyright 2024 Lupsha Egor
#pragma once

#include <vector>
#include <functional>
#include "core/task/include/task.hpp"

namespace lupsha_e_rect_integration_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void function_set(const std::function<double(double)>& func);
 private:
  double lower_bound{};
  double upper_bound{};
  int num_intervals{};
  std::vector<double> input_;
  std::vector<double> results_;
  std::function<double(double)> f;
};

}  // namespace lupsha_e_rect_integration_seq