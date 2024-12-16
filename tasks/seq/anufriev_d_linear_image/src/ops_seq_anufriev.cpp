#include "seq/anufriev_d_linear_image/include/ops_seq_anufriev.hpp"

namespace anufriev_d_linear_image {

bool SimpleIntSEQ::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  auto expected_matrix_size = static_cast<size_t>(rows * cols);

  return rows >= 3 && cols >= 3 && taskData->inputs_count[0] == expected_matrix_size &&
         taskData->outputs_count[0] == expected_matrix_size && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool SimpleIntSEQ::pre_processing() {
  internal_order_test();
  auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  input_data_.assign(matrix_data, matrix_data + matrix_size);

  int result_size = taskData->outputs_count[0];
  processed_data_.resize(result_size, 0);

  return true;
}

bool SimpleIntSEQ::run() {
  internal_order_test();
  applyGaussianFilter();
  return true;
}

bool SimpleIntSEQ::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(processed_data_.begin(), processed_data_.end(), output_data);
  return true;
}

void SimpleIntSEQ::applyGaussianFilter() {
  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      int sum = 0;
      for (int kr = -1; kr <= 1; ++kr) {
        for (int kc = -1; kc <= 1; ++kc) {
          sum += input_data_[(r + kr) * cols + (c + kc)] * kernel_[kr + 1][kc + 1];
        }
      }
      processed_data_[r * cols + c] = sum / 16;
    }
  }
}

}  // namespace anufriev_d_linear_image