// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace suvorov_d_shell_with_ord_merge_seq {

class TaskShellSortSeq : public ppc::core::Task {
 public:
  explicit TaskShellSortSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_to_sort;
  std::vector<int> sorted_data;

  static std::vector<int> shell_sort(const std::vector<int>&);
};

}  // namespace suvorov_d_shell_with_ord_merge_seq