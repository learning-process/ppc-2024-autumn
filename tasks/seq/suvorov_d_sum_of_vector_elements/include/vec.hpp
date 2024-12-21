// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace suvorov_d_sum_of_vector_elements_seq {

class Sum_of_vector_elements_seq : public ppc::core::Task {
 public:
  explicit Sum_of_vector_elements_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};

}  // namespace suvorov_d_sum_of_vector_elements_seq