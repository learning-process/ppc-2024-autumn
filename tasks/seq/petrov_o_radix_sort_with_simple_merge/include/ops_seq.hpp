#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_radix_sort_with_simple_merge_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace petrov_o_radix_sort_with_simple_merge_seq