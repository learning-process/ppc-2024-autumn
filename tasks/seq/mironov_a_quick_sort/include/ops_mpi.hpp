#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace mironov_a_quick_sort_seq {

class QuickSortSequential : public ppc::core::Task {
 public:
  explicit QuickSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> result_;
};

}  // namespace mironov_a_quick_sort_seq