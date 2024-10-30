// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_num_of_alternations_signs_seq {

using namespace std::chrono_literals;

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<int> input_{};
  int res{};
};

}  // namespace petrov_o_num_of_alternations_signs_seq