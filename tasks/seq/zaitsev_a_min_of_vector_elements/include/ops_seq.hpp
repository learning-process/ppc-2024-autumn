// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaitsev_a_min_of_vector_elements_seq {

class MinOfVectorElementsSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorElementsSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input{};
  int res{};
};

}  // namespace zaitsev_a_min_of_vector_elements_seq