// Copyright 2023 Nesterov Alexander
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_num_of_orderly_violations_seq {

template <class T>
class NumOfOrderlyViolations : public ppc::core::Task {
 private:
  std::vector<T> v;
  size_t n, res;

 public:
  explicit NumOfOrderlyViolations(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), n(taskData_->inputs_count[0]), res(0) {}
  bool count_num_of_orderly_violations_seq();
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_num_of_orderly_violations_seq