#include "seq/vasilev_s_gaus3x3/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool vasilev_s_gaus3x3_seq::Gaus3x3Sequential::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  int num_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  int num_cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  auto expected_matrix_size = static_cast<size_t>(num_rows * num_cols);

  return num_rows >= 3 && num_cols >= 3 && taskData->inputs_count[0] == expected_matrix_size &&
         taskData->outputs_count[0] == expected_matrix_size && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool vasilev_s_gaus3x3_seq::Gaus3x3Sequential::pre_processing() {
  internal_order_test();
  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  matrix.assign(matrix_data, matrix_data + matrix_size);

  int result_size = taskData->outputs_count[0];
  result_vector.resize(result_size, 0);

  return true;
}

bool vasilev_s_gaus3x3_seq::Gaus3x3Sequential::run() {
  internal_order_test();

  for (int row = 1; row < rows - 1; ++row) {
    for (int col = 1; col < cols - 1; ++col) {
      int baseIndex = row * cols + col;

      double result = 0.0;
      result += matrix[baseIndex - cols - 1] * 0.0625;  // Top-left
      result += matrix[baseIndex - cols] * 0.125;       // Top-center
      result += matrix[baseIndex - cols + 1] * 0.0625;  // Top-right
      result += matrix[baseIndex - 1] * 0.125;          // Middle-left
      result += matrix[baseIndex] * 0.25;               // Center
      result += matrix[baseIndex + 1] * 0.125;          // Middle-right
      result += matrix[baseIndex + cols - 1] * 0.0625;  // Bottom-left
      result += matrix[baseIndex + cols] * 0.125;       // Bottom-center
      result += matrix[baseIndex + cols + 1] * 0.0625;  // Bottom-right

      result_vector[baseIndex] = result;
    }
  }

  return true;
}

bool vasilev_s_gaus3x3_seq::Gaus3x3Sequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result_vector.begin(), result_vector.end(), output_data);

  return true;
}
