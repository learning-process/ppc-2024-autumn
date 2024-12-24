#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace gordeeva_t_shell_sort_batcher_merge_seq {

void shellSort(std::vector<int>& arr, int arr_length);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};

}  // namespace gordeeva_t_shell_sort_batcher_merge_seq