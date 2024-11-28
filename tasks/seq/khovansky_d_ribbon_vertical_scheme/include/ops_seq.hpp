// Copyright 2024 Khovansky Dmitry
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_ribbon_vertical_scheme_seq {

class RibbonVerticalSchemeSeq : public ppc::core::Task {
 public:
  explicit RibbonVerticalSchemeSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* hello_matrix;
  int* hello_vector;
  int rows_count{};
  int columns_count{};
  std::vector<int> goodbye_vector;
};
}  // namespace khovansky_d_ribbon_vertical_scheme_seq