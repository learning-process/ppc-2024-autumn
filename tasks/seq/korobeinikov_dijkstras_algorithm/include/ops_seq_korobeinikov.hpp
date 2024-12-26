// Copyright 2024 Korobeinikov Arseny
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace korobeinikov_a_test_task_seq_lab_03 {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  std::vector<int> res;

  int size;
  int sv;  // selected_vertex
};

}  // namespace korobeinikov_a_test_task_seq_lab_03