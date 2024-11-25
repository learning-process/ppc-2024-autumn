#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_{}, res{};
};

}  // namespace konkov_i_task_dining_philosophers