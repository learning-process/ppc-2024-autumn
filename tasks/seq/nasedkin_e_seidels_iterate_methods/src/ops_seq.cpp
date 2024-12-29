#include "seq/nasedkin_e_seidels_iterate_methods/include/ops_seq.hpp"

#include <cmath>
#include <iostream>

namespace nasedkin_e_seidels_iterate_methods_seq {

bool SeidelIterateMethodsSEQ::pre_processing() {
  if (!validation()) {
    return false;
  }

  epsilon = 1e-6;
  max_iterations = 1000;

  x.resize(n, 0.0);

  return taskData->inputs_count.size() <= 1 || taskData->inputs_count[1] != 0;
}

bool SeidelIterateMethodsSEQ::validation() {
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

bool SeidelIterateMethodsSEQ::run() {
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

bool SeidelIterateMethodsSEQ::post_processing() { return true; }

bool SeidelIterateMethodsSEQ::converge(const std::vector<double>& x_new) {
  double residual_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    double Ax_i = 0.0;
    for (int j = 0; j < n; ++j) {
      Ax_i += A[i][j] * x_new[j];
    }
    residual_norm += std::pow(Ax_i - b[i], 2);
  }
  return std::sqrt(residual_norm) < epsilon;
}

void SeidelIterateMethodsSEQ::set_matrix(const std::vector<std::vector<double>>& matrix,
                                         const std::vector<double>& vector) {
  A = matrix;
  b = vector;
  n = static_cast<int>(matrix.size());
}

void SeidelIterateMethodsSEQ::generate_random_matrix(int size, std::vector<std::vector<double>>& matrix,
                                                     std::vector<double>& vector) {
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

}  // namespace nasedkin_e_seidels_iterate_methods_seq