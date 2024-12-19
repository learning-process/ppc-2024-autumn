#include "seq/sarafanov_m_gauss_jordan_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

std::vector<double> sarafanov_m_gauss_jordan_method_seq::processMatrix(int n, int k,
                                                                       const std::vector<double>& matrix) {
  std::vector<double> result_vec(n * (n - k + 1));

  for (int i = 0; i < (n - k + 1); i++) {
    result_vec[i] = matrix[(n + 1) * k + k + i];
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * (i + 1) + j] = matrix[i * (n + 1) + k + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * i + j] = matrix[i * (n + 1) + k + j];
    }
  }

  return result_vec;
}

void sarafanov_m_gauss_jordan_method_seq::updateMatrix(int n, int k, std::vector<double>& matrix,
                                                       const std::vector<double>& iter_result) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[i * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[(i - 1) * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n + 1; i++) matrix[k * (n + 1) + i] /= matrix[k * (n + 1) + k];

  for (int i = 0; i < n; i++) {
    matrix[i * (n + 1) + k] = 0;
  }

  matrix[k * (n + 1) + k] = 1;
}

bool sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential::validation() {
  internal_order_test();
  int n_val = *reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_size = taskData->inputs_count[0];
  return n_val * (n_val + 1) == matrix_size;
}

bool sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential::pre_processing() {
  internal_order_test();

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];
  n = *reinterpret_cast<int*>(taskData->inputs[1]);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential::run() {
  internal_order_test();

  for (int k = 0; k < n; k++) {
    if (matrix[k * (n + 1) + k] == 0) {
      int change;
      for (change = k + 1; change < n; change++) {
        if (matrix[change * (n + 1) + k] != 0) {
          for (int col = 0; col < (n + 1); col++) {
            std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
          }
          break;
        }
      }
      if (change == n) return false;
    }

    std::vector<double> iter_matrix = sarafanov_m_gauss_jordan_method_seq::processMatrix(n, k, matrix);

    std::vector<double> iter_result((n - 1) * (n - k));

    int ind = 0;
    for (int i = 1; i < n; ++i) {
      for (int j = 1; j < n - k + 1; ++j) {
        double rel = iter_matrix[0];
        double nel = iter_matrix[i * (n - k + 1) + j];
        double a = iter_matrix[j];
        double b = iter_matrix[i * (n - k + 1)];
        double res = nel - (a * b) / rel;
        iter_result[ind++] = res;
      }
    }

    sarafanov_m_gauss_jordan_method_seq::updateMatrix(n, k, matrix, iter_result);
  }

  return true;
}

bool sarafanov_m_gauss_jordan_method_seq::GaussJordanMethodSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(matrix.begin(), matrix.end(), output_data);

  return true;
}
