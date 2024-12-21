#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace rams_s_radix_sort_with_simple_merge_for_doubles_seq {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input;
  std::vector<double> result;
};

}  // namespace rams_s_radix_sort_with_simple_merge_for_doubles_seq
