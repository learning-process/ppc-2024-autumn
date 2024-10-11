// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_vector_sum_seq {

class VectorSumSequential : public ppc::core::Task {
 public:
  explicit VectorSumSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_{}, res{};
};

}  // namespace titov_s_vector_sum_seq