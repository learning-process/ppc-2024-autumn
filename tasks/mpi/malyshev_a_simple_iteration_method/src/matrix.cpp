#include "mpi/malyshev_a_simple_iteration_method/include/matrix.hpp"

double malyshev_a_simple_iteration_method_mpi::determinant(const std::vector<std::vector<double>>& matrix) {
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

int malyshev_a_simple_iteration_method_mpi::rank(const std::vector<std::vector<double>>& matrix) {
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