// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_sum_cols_matrix/include/ops_seq.hpp"

#include <thread>

int createLinearCoordinates(int x, int y, int width) { return x * width + y; }

std::vector<int> computeMatrixSumSequential(const std::vector<int>& matrix, int width, int height, int startX,
                                            int endX) {
  std::vector<int> sums;
  for (int col = startX; col < endX; ++col) {
    int totalColumnSum = 0;
    for (int row = 0; row < height; ++row) {
      int index = createLinearCoordinates(col, row, width);
      totalColumnSum += matrix[index];
    }
    sums.push_back(totalColumnSum);
  }
  return sums;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Prepare input and output data structures
  inputData_ = std::vector<int>(taskData->inputs_count[0]);
  int* dataPointer = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; ++i) {
    inputData_[i] = dataPointer[i];
  }
  numCols = taskData->inputs_count[1];
  numRows = taskData->inputs_count[2];
  resultData_ = std::vector<int>(numCols, 0);
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::validation() {
  internal_order_test();
  // Ensure output element count matches expectations
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::run() {
  internal_order_test();
  resultData_ = computeMatrixSumSequential(inputData_, numCols, numRows, 0, numCols);
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < numCols; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = resultData_[i];
  }
  return true;
}
