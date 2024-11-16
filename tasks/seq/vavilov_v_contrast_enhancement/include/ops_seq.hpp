#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_contrast_enhancement_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  int p_min_{0}, p_max_{255};
};

}  // namespace vavilov_v_contrast_enhancement_seq
