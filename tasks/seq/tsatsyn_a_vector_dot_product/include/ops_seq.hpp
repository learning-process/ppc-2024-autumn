// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_vector_dot_product_seq {
int resulting(const std::vector<int>& v1, const std::vector<int>& v2);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> v1;
  std::vector<int> v2;
  int res{};
};

}  // namespace tsatsyn_a_vector_dot_product_seq