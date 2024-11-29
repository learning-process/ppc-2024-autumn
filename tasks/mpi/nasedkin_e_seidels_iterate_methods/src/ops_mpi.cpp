#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>

namespace nasedkin_e_seidels_iterate_methods_mpi {

bool SeidelIterateMethodsMPI::pre_processing() {
  if (!validation()) {
    return false;
  }

  epsilon = 1e-6;
  max_iterations = 1000;

  x.assign(n, 0.0);

  return taskData->inputs_count.size() <= 1 || taskData->inputs_count[1] != 0;
}

bool SeidelIterateMethodsMPI::validation() {
  if (taskData->inputs_count.empty()) {
    return false;
  }

  n = taskData->inputs_count[0];
  if (n <= 0) {
    return false;
  }

  A.resize(n, std::vector<double>(n, 0.0));
  b.resize(n, 0.0);

  bool zero_diagonal_test = false;
  if (taskData->inputs_count.size() > 1 && taskData->inputs_count[1] == 0) {
    zero_diagonal_test = true;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A[i][j] = (i != j) ? 1.0 : 0.0;
      }
      b[i] = 1.0;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A[i][j] = (i == j) ? 2.0 : 1.0;
      }
      b[i] = n + 1;
    }
  }

  for (int i = 0; i < n; ++i) {
    if (A[i][i] == 0.0 && !zero_diagonal_test) {
      return false;
    }
  }

  return true;
}

bool SeidelIterateMethodsMPI::run() {
  std::vector<double> x_new(n, 0.0);
  int iteration = 0;

  while (iteration < max_iterations) {
    for (int i = 0; i < n; ++i) {
      x_new[i] = b[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          x_new[i] -= A[i][j] * x[j];
        }
      }
      if (A[i][i] == 0.0) {
        throw std::runtime_error("Diagonal element of A is zero, cannot proceed with division.");
      }
      x_new[i] /= A[i][i];
    }

    if (converge(x_new)) {
      x = x_new;
      return true;
    }

    x = x_new;
    ++iteration;
  }

  std::cerr << "Warning: Seidel method did not converge within max_iterations." << std::endl;
  return false;
}

bool SeidelIterateMethodsMPI::post_processing() { return true; }

bool SeidelIterateMethodsMPI::converge(const std::vector<double>& x_new) {
  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
  }
  return std::sqrt(norm) < epsilon;
}

void SeidelIterateMethodsMPI::set_matrix(const std::vector<std::vector<double>>& matrix,
                                         const std::vector<double>& vector) {
  if (matrix.size() != vector.size() || matrix.empty()) {
    throw std::invalid_argument("Matrix and vector dimensions do not match or are empty.");
  }
  A = matrix;
  b = vector;
  n = static_cast<int>(matrix.size());
}

void SeidelIterateMethodsMPI::generate_random_diag_dominant_matrix(int size, std::vector<std::vector<double>>& matrix, std::vector<double>& vector) {
    matrix.resize(size, std::vector<double>(size, 0.0));
    vector.resize(size, 0.0);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int i = 0; i < size; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < size; ++j) {
            if (i != j) {
                matrix[i][j] = static_cast<double>(std::rand() % 10 + 1);
                row_sum += std::abs(matrix[i][j]);
            }
        }
        matrix[i][i] = row_sum + static_cast<double>(std::rand() % 5 + 1);
        vector[i] = static_cast<double>(std::rand() % 20 + 1);
    }
}

double nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::check_residual_norm() const {
  if (A.empty() || b.empty() || x.empty()) {
    throw std::runtime_error("Matrix, vector, or solution vector is empty.");
  }

  double norm = 0.0;

  for (int i = 0; i < n; ++i) {
    double row_residual = b[i];
    for (int j = 0; j < n; ++j) {
      row_residual -= A[i][j] * x[j];
    }

    if (std::isnan(row_residual) || std::isinf(row_residual)) {
      throw std::runtime_error("Residual computation resulted in NaN or Inf.");
    }

    norm += row_residual * row_residual;
  }

  return std::sqrt(norm);
}

}  // namespace nasedkin_e_seidels_iterate_methods_mpi