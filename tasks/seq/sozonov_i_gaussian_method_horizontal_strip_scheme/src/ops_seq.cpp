#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    matrix[i] = tmp_ptr[i];
  }
  n = taskData->inputs_count[1];
  // Init value for output
  x = std::vector<double>(n, 0);
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] == taskData->inputs_count[1] * (taskData->inputs_count[1] + 1) &&
         taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < n - 1; ++i) {
    for (size_t k = i + 1; k < n; ++k) {
      double m = matrix[k * (n + 1) + i] / matrix[i * (n + 1) + i];
      for (size_t j = i; j < (n + 1); ++j) {
        matrix[k * (n + 1) + j] -= matrix[i * (n + 1) + j] * m;
      }
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double b = matrix[i * (n + 1) + n];
    for (size_t j = i + 1; j < n; ++j) {
      b -= matrix[i * (n + 1) + j] * x[j];
    }
    x[i] = b / matrix[i * (n + 1) + i];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < n; ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}