#include "seq/petrov_o_horizontal_gauss_method/include/ops_seq.hpp"

#include <algorithm>

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
    return false;
  }

  size_t n = taskData->inputs_count[0];

  if (n == 0) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);

  std::vector<double> valid_matrix(matrix_input, matrix_input + n * n);

  int rank_matrix = 0;

  const double eps = 1e-10;

  for (size_t i = 0; i < n; ++i) {
    double max_elem = std::abs(valid_matrix[i * n + i]);
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(valid_matrix[k * n + i]) > max_elem) {
        max_elem = std::abs(valid_matrix[k * n + i]);
        max_row = k;
      }
    }

    if (max_elem < eps) {
      continue;
    }

    if (max_row != i) {
      for (size_t j = 0; j < n; ++j) {
        std::swap(valid_matrix[i * n + j], valid_matrix[max_row * n + j]);
      }
    }

    double pivot = valid_matrix[i * n + i];
    for (size_t j = i; j < n; ++j) {
      valid_matrix[i * n + j] /= pivot;
    }

    for (size_t k = i + 1; k < n; ++k) {
      double factor = valid_matrix[k * n + i];
      for (size_t j = i; j < n; ++j) {
        valid_matrix[k * n + j] -= factor * valid_matrix[i * n + j];
      }
    }

    ++rank_matrix;
  }

  return rank_matrix == static_cast<int>(n);
}

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::pre_processing() {
  internal_order_test();

  size_t n = taskData->inputs_count[0];

  matrix.resize(n * n);
  b.resize(n);
  x.resize(n);

  auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(matrix_input, matrix_input + n * n, matrix.begin());

  auto* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(b_input, b_input + n, b.begin());

  return true;
}

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::run() {
  internal_order_test();

  size_t n = taskData->inputs_count[0];

  for (size_t k = 0; k < n - 1; ++k) {
    for (size_t i = k + 1; i < n; ++i) {
      double factor = matrix[i * n + k] / matrix[k * n + k];
      for (size_t j = k; j < n; ++j) {
        matrix[i * n + j] -= factor * matrix[k * n + j];
      }
      b[i] -= factor * b[k];
    }
  }

  x[n - 1] = b[n - 1] / matrix[(n - 1) * n + (n - 1)];
  for (int i = n - 2; i >= 0; --i) {
    double sum = b[i];
    for (size_t j = i + 1; j < n; ++j) {
      sum -= matrix[i * n + j] * x[j];
    }
    x[i] = sum / matrix[i * n + i];
  }

  return true;
}

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::post_processing() {
  internal_order_test();

  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x.begin(), x.end(), output);
  return true;
}