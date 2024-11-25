#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_vertical_ribbon_scheme_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int num_rows_;
  int num_cols_;
};

}  // namespace sedova_o_vertical_ribbon_scheme_seq