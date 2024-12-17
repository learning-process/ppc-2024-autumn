// Copyright 2024 Nesterov Alexander
#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_a_Shell_sort_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_;

  void compareAndSwap(int i, int j);
};

}  // namespace petrov_a_Shell_sort_seq
