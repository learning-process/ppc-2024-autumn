#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_vertical_ribbon_scheme_seq {

class Sequential : public ppc::core::Task {
 public:
  explicit Sequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* matrix_;
  int* vector_;
  std::vector<std::vector<int>> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int count;
  int rows_;
  int cols_;
};

}  // namespace sedova_o_vertical_ribbon_scheme_seq