// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_max_col_matrix_seq {

std::vector<std::vector<int32_t>> getRandomMatrix(uint32_t row, uint32_t col);
void insertRefValue(std::vector<std::vector<int32_t>>& mtrx, int32_t ref);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> input_;
  std::vector<int32_t> res_;
};

}  // namespace kondratev_ya_max_col_matrix_seq