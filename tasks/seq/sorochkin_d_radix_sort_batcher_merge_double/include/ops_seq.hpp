#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace sorochkin_d_radix_sort_batcher_merge_double_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_, res_;
};

}  // namespace sorochkin_d_radix_sort_batcher_merge_double_seq