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

  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      A[i][j] = (i == j) ? static_cast<double>(std::rand() % 10 + 1) : static_cast<double>(std::rand() % 5);
      row_sum += (i != j) ? A[i][j] : 0.0;
    }
    A[i][i] = row_sum + 1.0;
    b[i] = static_cast<double>(std::rand() % 10);
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
      break;
    }

    x = x_new;
    ++iteration;
  }

  double error_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    double Ax_i = 0.0;
    for (int j = 0; j < n; ++j) {
      Ax_i += A[i][j] * x[j];
    }
    error_norm += (Ax_i - b[i]) * (Ax_i - b[i]);
  }

  error_norm = std::sqrt(error_norm);
  if (error_norm >= epsilon) {
    std::cerr << "Error norm ||Ax - b|| = " << error_norm << " is not less than epsilon" << std::endl;
    return false;
  }

  return true;
}

bool SeidelIterateMethodsMPI::post_processing() { return true; }

bool SeidelIterateMethodsMPI::converge(const std::vector<double>& x_new) {
  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
  }
  return std::sqrt(norm) < epsilon;
}

}  // namespace nasedkin_e_seidels_iterate_methods_mpi