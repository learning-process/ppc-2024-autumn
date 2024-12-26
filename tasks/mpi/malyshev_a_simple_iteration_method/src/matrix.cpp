#include "mpi/malyshev_a_simple_iteration_method/include/matrix.hpp"

double malyshev_a_simple_iteration_method_mpi::determinant(const std::vector<double>& matrix, uint32_t n) {
  auto local_matrix(matrix);
  double det = 1.0;

  for (uint32_t i = 0; i < n - 1; ++i) {
    for (uint32_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j * n + i] / local_matrix[i * n + i];
      for (uint32_t k = i; k < n; ++k) {
        local_matrix[j * n + k] -= factor * local_matrix[i * n + k];
      }
    }
  }

  for (uint32_t i = 0; i < n; ++i) {
    det *= local_matrix[i * n + i];
  }

  return det;
}

int malyshev_a_simple_iteration_method_mpi::rank(const std::vector<double>& matrix, uint32_t n, uint32_t m) {
  auto local_matrix(matrix);
  int rank = 0;

  for (uint32_t i = 0; i < std::min(n, m); ++i) {
    uint32_t max_row = i;
    for (uint32_t k = i + 1; k < n; ++k) {
      if (std::abs(local_matrix[k * m + i]) > std::abs(local_matrix[max_row * m + i])) {
        max_row = k;
      }
    }

    if (std::abs(local_matrix[max_row * m + i]) < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    std::swap(local_matrix[i], local_matrix[max_row]);

    for (uint32_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j * m + i] / local_matrix[i * m + i];
      for (uint32_t k = i; k < m; ++k) {
        local_matrix[j * m + k] -= factor * local_matrix[i * m + k];
      }
    }
    rank++;
  }

  return rank;
}