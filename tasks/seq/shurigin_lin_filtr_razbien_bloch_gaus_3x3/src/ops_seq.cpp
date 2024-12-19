#include "seq/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_seq.hpp"

bool shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  int total_rows = *reinterpret_cast<int*>(taskData->inputs[1]);
  int total_cols = *reinterpret_cast<int*>(taskData->inputs[2]);

  if (total_rows < 3 || total_cols < 3) {
    return false;
  }

  auto expected_size = static_cast<size_t>(total_rows * total_cols);
  if (taskData->inputs_count[0] != expected_size || taskData->outputs_count[0] != expected_size ||
      taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  for (size_t i = 0; i < expected_size; ++i) {
    if (matrix_data[i] < 0.0 || matrix_data[i] > 255.0) {
      return false;
    }
  }

  return true;
}

bool shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq::pre_processing() {
  internal_order_test();

  num_rows = *reinterpret_cast<const int*>(taskData->inputs[1]);
  num_cols = *reinterpret_cast<const int*>(taskData->inputs[2]);

  const auto* source_pixels = reinterpret_cast<const double*>(taskData->inputs[0]);
  const size_t total_pixels = taskData->inputs_count[0];

  data_matrix = std::vector<double>(source_pixels, source_pixels + total_pixels);

  output_vector = std::vector<double>(taskData->outputs_count[0]);
  std::fill(output_vector.begin(), output_vector.end(), 0.0);

  return true;
}

bool shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq::run() {
  internal_order_test();

  for (int r = 1; r < num_rows - 1; ++r) {
    for (int c = 1; c < num_cols - 1; ++c) {
      const int pos = r * num_cols + c;
      double sum = 0.0;

      sum = sum + data_matrix[pos - num_cols - 1] * 0.0625;
      sum = sum + data_matrix[pos - num_cols] * 0.125;
      sum = sum + data_matrix[pos - num_cols + 1] * 0.0625;
      sum = sum + data_matrix[pos - 1] * 0.125;
      sum = sum + data_matrix[pos] * 0.25;
      sum = sum + data_matrix[pos + 1] * 0.125;
      sum = sum + data_matrix[pos + num_cols - 1] * 0.0625;
      sum = sum + data_matrix[pos + num_cols] * 0.125;
      sum = sum + data_matrix[pos + num_cols + 1] * 0.0625;

      output_vector[pos] = sum;
    }
  }

  return true;
}

bool shurigin_lin_filtr_razbien_bloch_gaus_3x3::TaskSeq::post_processing() {
  internal_order_test();

  auto* result_data = reinterpret_cast<double*>(taskData->outputs[0]);
  const size_t size = output_vector.size();

  for (size_t i = 0; i < size; ++i) {
    result_data[i] = output_vector[i];
  }

  return true;
}