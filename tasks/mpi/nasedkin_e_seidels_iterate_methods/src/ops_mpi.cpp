#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

#include <cmath>
#include <iostream>
#include <random>

namespace nasedkin_e_seidels_iterate_methods_mpi {

bool SeidelIterateMethodsMPI::pre_processing() {
  if (!validation()) {
    return false;
  }

  epsilon = 1e-6;
  max_iterations = 1000;

  x.resize(n, 0.0);

  return true;
}

bool SeidelIterateMethodsMPI::validation() {
  if (taskData->inputs_count.empty()) {
    return false;
  }

  n = taskData->inputs_count[0];
  if (n <= 0) {
    return false;
  }

  generate_valid_matrix();
  return true;
}

void SeidelIterateMethodsMPI::generate_valid_matrix() {
  A.resize(n, std::vector<double>(n, 0.0));
  b.resize(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i][j] = dist(gen);
        sum += std::abs(A[i][j]);
      }
    }
    A[i][i] = sum + std::abs(dist(gen)) + 1.0;
    b[i] = dist(gen);
  }
}

bool SeidelIterateMethodsMPI::run() {
  std::vector<double> x_new = x;
  int iteration = 0;

  while (iteration < max_iterations) {
    for (int i = 0; i < n; ++i) {
      double sigma = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sigma += A[i][j] * x[j];
        }
      }
      x_new[i] = (b[i] - sigma) / A[i][i];
    }

    if (converge(x_new)) {
      x = x_new;
      std::cout << "Converged in " << iteration << " iterations.\n";
      return true;
    }

    x.swap(x_new);
    ++iteration;
  }

  std::cout << "Failed to converge after " << max_iterations << " iterations.\n";
  return compute_residual_norm() < epsilon;
}



double SeidelIterateMethodsMPI::compute_residual_norm() {
  std::vector<double> residual(n, 0.0);
  double norm = 0.0;

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += A[i][j] * x[j];
    }
    residual[i] = sum - b[i];
    norm += residual[i] * residual[i];
  }

  return std::sqrt(norm);
}

bool SeidelIterateMethodsMPI::post_processing() { return true; }

bool SeidelIterateMethodsMPI::converge(const std::vector<double>& x_new) {
  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += std::pow(x_new[i] - x[i], 2);
  }
  return std::sqrt(norm) < epsilon;
}

}  // namespace nasedkin_e_seidels_iterate_methods_mpi
