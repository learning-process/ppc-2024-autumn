#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_max_column_task_seq {

class MaxColumnTaskSequential : public ppc::core::Task {
 public:
  explicit MaxColumnTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> res;
  std::vector<int> input_;
  int row_count{};
  int column_count{};
};

}  // namespace kapustin_i_max_column_task_seq