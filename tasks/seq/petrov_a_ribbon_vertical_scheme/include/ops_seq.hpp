// Copyright 2024 Nesterov Alexander
#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_a_ribbon_vertical_scheme_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> matrix_;
  std::vector<int> vector_;
  std::vector<int> result_;
};

}  // namespace petrov_a_ribbon_vertical_scheme_seq
