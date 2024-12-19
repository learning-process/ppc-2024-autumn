// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace matyunina_a_average_of_vector_elements_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};

}  // namespace matyunina_a_average_of_vector_elements_seq