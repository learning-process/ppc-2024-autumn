#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace sarafanov_m_quick_sort_batcher_merge_seq {

const int THRESHOLD = 16;

void batcher_merge(std::vector<int>& arr, int left, int right);
void quick_sort(std::vector<int>& arr, int left, int right);
void quick_sort_with_batcher_merge(std::vector<int>& arr);

class QuicksortBatcherMerge : public ppc::core::Task {
 public:
  explicit QuicksortBatcherMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> arr;
};

}  // namespace sarafanov_m_quick_sort_batcher_merge_seq
