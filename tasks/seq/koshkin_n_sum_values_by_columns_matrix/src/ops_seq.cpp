#include "seq/koshkin_n_sum_values_by_columns_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];

  // TaskData
  input_.resize(rows, std::vector<int>(columns));

  int* inputMatrix = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      input_[i][j] = inputMatrix[i * columns + j];
    }
  }

  res.resize(columns, 0);  // sumColumns
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < columns; ++j) {
    res[j] = 0;
    for (int i = 0; i < rows; ++i) {
      res[j] += input_[i][j];
    }
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* outputSums = reinterpret_cast<int*>(taskData->outputs[0]);
  for (int j = 0; j < columns; ++j) {
    outputSums[j] = res[j];
  }
  return true;
}