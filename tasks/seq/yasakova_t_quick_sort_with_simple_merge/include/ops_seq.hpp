#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_quick_sort_with_simple_merge_seq {

class QuickSortWithMergeSeq : public ppc::core::Task {
 public:
  explicit QuickSortWithMergeSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_vector;
};

}  // namespace yasakova_t_quick_sort_with_simple_merge_seq