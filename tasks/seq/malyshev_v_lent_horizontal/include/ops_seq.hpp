#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_lent_horizontal {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;
};

}  // namespace malyshev_lent_horizontal