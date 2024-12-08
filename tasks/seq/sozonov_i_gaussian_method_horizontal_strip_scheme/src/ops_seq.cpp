#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

int sozonov_i_gaussian_method_horizontal_strip_scheme_seq::extended_matrix_rank(int n, int m, std::vector<double> a) {
  const double EPS = 1e-9;

  int rank = m;
  for (int i = 0; i < m; ++i) {
    int j;
    for (j = 0; j < n; ++j) {
      if (std::abs(a[j * n + i]) > EPS) {
        break;
      }
    }
    if (j == n) {
      --rank;
    } else {
      for (int k = i + 1; k < m; ++k) {
        double ml = a[k * n + i] / a[i * n + i];
        for (j = i; j < n - 1; ++j) {
          a[k * n + j] -= a[i * n + j] * ml;
        }
      }
    }
  }
  return rank;
}

int sozonov_i_gaussian_method_horizontal_strip_scheme_seq::determinant(int n, int m, std::vector<double> a) {
  const double EPS = 1e-9;
  double det = 1;

  for (int i = 0; i < m; ++i) {
    int idx = i;
    for (int k = i + 1; k < m; ++k) {
      if (std::abs(a[k * n + i]) > std::abs(a[idx * n + i])) {
        idx = k;
      }
    }
    if (std::abs(a[idx * n + i]) < EPS) {
      return 0;
    }
    if (idx != i) {
      for (int j = 0; j < n - 1; ++j) {
        double tmp = a[i * n + j];
        a[i * n + j] = a[idx * n + j];
        a[idx * n + j] = tmp;
      }
      det *= -1;
    }
    det *= a[i * n + i];
    for (int k = i + 1; k < m; ++k) {
      double ml = a[k * n + i] / a[i * n + i];
      for (int j = i; j < n - 1; ++j) {
        a[k * n + j] -= a[i * n + j] * ml;
      }
    }
  }
  return det;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];
  // Init value for output
  x = std::vector<double>(cols - 1, 0);
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];

  // Check matrix for a single solution
  return taskData->inputs_count[0] > 0 && rows == cols - 1 && determinant(cols, rows, matrix) != 0 &&
         extended_matrix_rank(cols, rows, matrix) == rows;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows - 1; ++i) {
    for (int k = i + 1; k < rows; ++k) {
      double m = matrix[k * cols + i] / matrix[i * cols + i];
      for (int j = i; j < cols; ++j) {
        matrix[k * cols + j] -= matrix[i * cols + j] * m;
      }
    }
  }
  for (int i = rows - 1; i >= 0; --i) {
    double sum = matrix[i * cols + rows];
    for (int j = i + 1; j < cols - 1; ++j) {
      sum -= matrix[i * cols + j] * x[j];
    }
    x[i] = sum / matrix[i * cols + i];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cols - 1; ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}