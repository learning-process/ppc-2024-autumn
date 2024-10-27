// Copyright 2024 Korobeinikov Arseny
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace korobeinikov_a_test_task_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
 private:
  std::vector<int> input_;
  std::vector<int> res;
  int count_rows{};
  int size_rows{};
};

}  // namespace korobeinikov_a_test_task_seq