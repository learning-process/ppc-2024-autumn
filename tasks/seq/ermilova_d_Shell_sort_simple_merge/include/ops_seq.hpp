// Copyright 2023 Nesterov Alexander
#pragma once

#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermilova_d_Shell_sort_simple_merge_seq {

std::vector<int> ShellSort(std::vector<int>& vec, const std::function<bool(int, int)>& comp);

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
  bool is_descending{};
};

}  // namespace ermilova_d_Shell_sort_simple_merge_seq