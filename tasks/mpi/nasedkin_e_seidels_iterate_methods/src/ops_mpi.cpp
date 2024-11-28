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

  A.resize(n, std::vector<double>(n, 0.0));
  b.resize(n, 0.0);
  x.resize(n, 0.0);

  if (taskData->inputs_count.size() > 1 && taskData->inputs_count[1] == 0) {
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
    if (A[i][i] == 0.0) {
      return false;
    }
  }
  return true;


}

bool SeidelIterateMethodsMPI::validation() {
  if (taskData->inputs_count.empty()) {
    return false;
  }

  n = taskData->inputs_count[0];
  return n > 0;
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

void nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_system(int size, double min_val, double max_val) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    A.resize(size, std::vector<double>(size));
    b.resize(size);
    x.resize(size, 0.0);

    for (int i = 0; i < size; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < size; ++j) {
            A[i][j] = dist(gen);
            if (i != j) row_sum += std::abs(A[i][j]);
        }
        A[i][i] = row_sum + dist(gen);
        b[i] = dist(gen);
    }
}

double nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::compute_residual_norm(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& x,
    const std::vector<double>& b) {
    std::vector<double> Ax_b(b.size(), 0.0);

    for (size_t i = 0; i < A.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < A[i].size(); ++j) {
            sum += A[i][j] * x[j];
        }
        Ax_b[i] = sum - b[i];
    }

    double norm = 0.0;
    for (double val : Ax_b) {
        norm += val * val;
    }
    return std::sqrt(norm);
}

}  // namespace nasedkin_e_seidels_iterate_methods_mpi