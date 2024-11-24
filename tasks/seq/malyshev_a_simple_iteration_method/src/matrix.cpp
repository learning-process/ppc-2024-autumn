#include "seq/malyshev_a_simple_iteration_method/include/matrix.hpp"

double malyshev_a_simple_iteration_method_seq::determinant(const std::vector<std::vector<double>>& matrix) {
  auto local_matrix(matrix);
  uint32_t n = local_matrix.size();
  double det = 1.0;

  for (uint32_t i = 0; i < n - 1; ++i) {
    for (uint32_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j][i] / local_matrix[i][i];
      for (uint32_t k = i; k < n; ++k) {
        local_matrix[j][k] -= factor * local_matrix[i][k];
      }
    }
  }

  for (uint32_t i = 0; i < n; ++i) {
    det *= local_matrix[i][i];
  }

  return det;
}

double malyshev_a_simple_iteration_method_seq::determinant(const std::vector<double>& matrix, uint32_t n) {
  std::vector<std::vector<double>> tmp(n, std::vector<double>(n));
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++) {
      tmp[i][j] = matrix[i * n + j];
    }
  }
  return determinant(tmp);
}

int malyshev_a_simple_iteration_method_seq::rank(const std::vector<std::vector<double>>& matrix) {
  auto local_matrix(matrix);
  uint32_t n = local_matrix.size();
  uint32_t m = local_matrix[0].size();
  int rank = 0;

  for (uint32_t i = 0; i < std::min(n, m); ++i) {
    uint32_t max_row = i;
    for (uint32_t k = i + 1; k < n; ++k) {
      if (std::abs(local_matrix[k][i]) > std::abs(local_matrix[max_row][i])) {
        max_row = k;
      }
    }

    if (std::abs(local_matrix[max_row][i]) < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    std::swap(local_matrix[i], local_matrix[max_row]);

    for (uint32_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j][i] / local_matrix[i][i];
      for (uint32_t k = i; k < m; ++k) {
        local_matrix[j][k] -= factor * local_matrix[i][k];
      }
    }
    rank++;
  }

  return rank;
}

void malyshev_a_simple_iteration_method_seq::getRandomData(uint32_t n, std::vector<double>& A, std::vector<double>& B) {
  const auto random_double = [&](double lower_bound, double upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * (std::rand() % RAND_MAX) / RAND_MAX;
  };

  std::srand(std::time(nullptr));

  std::vector<double> X(n);
  for (uint32_t i = 0; i < n; i++) {
    X[i] = random_double(50, 100);
    if (std::rand() % 2 == 0) X[i] *= -1;
  }

  A.resize(n * n);
  B.resize(n);

  double sum_by_row_for_C;
  double sum_by_row_for_B;
  for (uint32_t i = 0; i < n; i++) {
    A[i * n + i] = random_double(50, 100);
    if (std::rand() % 2 == 0) A[i * n + i] *= -1;

    sum_by_row_for_C = 0.01;
    sum_by_row_for_B = A[i * n + i] * X[i];

    for (uint32_t j = 0; j < n; j++) {
      if (i != j) {
        A[i * n + j] =
            random_double(std::abs(A[i * n + i]) * (-1 + sum_by_row_for_C + std::numeric_limits<double>::epsilon()),
                          std::abs(A[i * n + i]) * (1 - sum_by_row_for_C - std::numeric_limits<double>::epsilon()));

        sum_by_row_for_C += std::abs(A[i * n + j] / A[i * n + i]);
        sum_by_row_for_B += A[i * n + j] * X[j];
      }
    }

    B[i] = sum_by_row_for_B;
  }
}