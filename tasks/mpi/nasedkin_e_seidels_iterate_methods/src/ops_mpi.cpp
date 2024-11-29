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

  x.resize(n, 0.0);

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

  if (taskData->inputs_count.size() > 1 && taskData->inputs_count[1] == 0) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A[i][j] = (i != j) ? 1.0 : 0.0;
      }
      b[i] = 1.0;
    }
    return true;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i][j] = (i == j) ? 2.0 : 1.0;
    }
    b[i] = n + 1;
  }

  for (int i = 0; i < n; ++i) {
    if (A[i][i] == 0.0) {
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
      x_new[i] /= A[i][i];
    }

    if (converge(x_new)) {
      x = x_new;
      return true;
    }

    x = x_new;
    ++iteration;
  }

  return false;
}

bool SeidelIterateMethodsMPI::post_processing() {
  return true;
}

bool SeidelIterateMethodsMPI::converge(const std::vector<double>& x_new) {
  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
  }
  norm = std::sqrt(norm);

  return norm < epsilon;
}

void SeidelIterateMethodsMPI::set_matrix(const std::vector<std::vector<double>>& matrix,
                                         const std::vector<double>& vector) {
  if (matrix.empty() || matrix.size() != vector.size()) {
    std::cerr << "Matrix and vector dimensions do not match or are empty." << std::endl;
    return;
  }

  for (size_t i = 0; i < matrix.size(); ++i) {
    if (matrix[i].size() != matrix.size()) {
      std::cerr << "Matrix must be square." << std::endl;
      return;
    }
    if (matrix[i][i] == 0.0) {
      std::cerr << "Matrix contains zero on the diagonal." << std::endl;
      return;
    }
  }

  A = matrix;
  b = vector;
  n = static_cast<int>(matrix.size());
}

void SeidelIterateMethodsMPI::generate_random_diag_dominant_matrix(int size, std::vector<std::vector<double>>& matrix,
                                                                   std::vector<double>& vector) {
  if (size <= 0) {
    std::cerr << "Matrix size must be positive." << std::endl;
    return;
  }

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

}  // namespace nasedkin_e_seidels_iterate_methods_mpi
