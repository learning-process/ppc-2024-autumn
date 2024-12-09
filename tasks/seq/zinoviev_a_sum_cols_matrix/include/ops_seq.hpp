// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

std::vector<int> computeMatrixSumSequential(const std::vector<int>& matrix, int width, int height, int startX,
                                            int endX);
int createLinearCoordinates(int x, int y, int width);

namespace zinoviev_a_sum_cols_matrix {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int numRows{};
  int numCols{};
  std::vector<int> inputData_;
  std::vector<int> resultData_;
};

}  // namespace zinoviev_a_sum_cols_matrix