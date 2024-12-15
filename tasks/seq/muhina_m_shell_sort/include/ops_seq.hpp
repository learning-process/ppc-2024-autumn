// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

namespace muhina_m_shell_sort_seq {
std::vector<int> shellSort(const std::vector<int>& vect);

class ShellSortSequential : public ppc::core::Task {
 public:
  explicit ShellSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};
}  // namespace muhina_m_shell_sort_seq