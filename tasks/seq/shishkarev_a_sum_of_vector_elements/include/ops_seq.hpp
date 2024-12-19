// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_sum_of_vector_elements_seq {
template <class InOutType>
class VectorSumSequential : public ppc::core::Task {
 public:
  explicit VectorSumSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<InOutType> input_data;
  InOutType result;
};

}  // namespace shishkarev_a_sum_of_vector_elements_seq
