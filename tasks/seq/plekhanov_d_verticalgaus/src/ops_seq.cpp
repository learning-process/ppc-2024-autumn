#include "seq/plekhanov_d_verticalgaus/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool plekhanov_d_verticalgaus_seq::VerticalGausSequential::pre_processing() {
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

bool plekhanov_d_verticalgaus_seq::VerticalGausSequential::validation() {
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

bool plekhanov_d_verticalgaus_seq::VerticalGausSequential::run() {
  internal_order_test();

  // Define Gaussian kernel weights
  const double weights[3][3] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};

  // Sum values for each square of Gaussian mask and then put new value in our new image. We need to ignore border
  // pixels
  for (int row = 1; row < rows - 1; row++) {
    for (int col = 1; col < cols - 1; col++) {
      double result = 0.0;
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int currentPixel = (row + i) * cols + (col + j);
          result += matrix[currentPixel] * weights[i + 1][j + 1];
        }
      }
      result_vector[row * cols + col] = result;
    }
  }

  return true;
}

bool plekhanov_d_verticalgaus_seq::VerticalGausSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result_vector.begin(), result_vector.end(), output_data);

  return true;
}
