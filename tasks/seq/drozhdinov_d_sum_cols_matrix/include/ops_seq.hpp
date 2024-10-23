// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

int makeLinCoords(int x, int y, int xSize);
std::vector<int> calculateMatrixSumSequentially(const std::vector<int>& matrix, int xSize, int ySize);
std::vector<int> calcMatrixSumSeq(const std::vector<int>& matrix, int xSize, int ySize, int fromX, int toX);
namespace drozhdinov_d_sum_cols_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows{};
  int cols{};
  std::vector<int> input_;
  std::vector<int> res;
};

}  // namespace drozhdinov_d_sum_cols_matrix_seq