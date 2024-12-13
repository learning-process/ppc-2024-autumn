// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalyakina_a_average_value_seq {

class FindingAverageOfVectorElementsTaskSequential : public ppc::core::Task {
 public:
  explicit FindingAverageOfVectorElementsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  double average_value{};
};

}  // namespace kalyakina_a_average_value_seq