// Copyright 2024 Tselikova Arina
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace tselikova_a_average_of_vector_elements {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_{};
  float res{};
};

}  // namespace tselikova_a_average_of_vector_elements