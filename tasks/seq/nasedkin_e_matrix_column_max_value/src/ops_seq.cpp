#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>

bool nasedkin_e_matrix_column_max_value_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  numCols = (int)*taskData->inputs[1];
  numRows = (int)(taskData->inputs_count[0] / numCols);
  inputMatrix_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmpPtr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    inputMatrix_[i] = tmpPtr[i];
  }

  result_ = std::vector<int>(numCols, 0);

  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::TestTaskSequential::validation() {
  internal_order_test();
  if ((int)*taskData->inputs[1] == 0) {
    return false;
  }
  if (taskData->inputs.empty() || taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (*taskData->inputs[1] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < numCols; j++) {
    auto columnStart = inputMatrix_.begin() + j;
    auto columnEnd = inputMatrix_.begin() + j + numRows * numCols;
    auto maxElement = *std::max_element(columnStart, columnEnd,
                                        [this](int a, int b) {
                                          return (a < b) && ((&a - inputMatrix_.data()) % numCols == 0);
                                        });
    result_[j] = maxElement;
  }

  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  for (int j = 0; j < numCols; j++) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = result_[j];
  }

  return true;
}