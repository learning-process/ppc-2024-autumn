#include "seq/korablev_v_jacobi_method/include/ops_seq.hpp"

#include <cmath>
#include <iostream>

bool korablev_v_jacobi_method_seq::JacobiMethodSequential::pre_processing() {
  internal_order_test();
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  A_.resize(n, std::vector<double>(n));
  b_.resize(n);
  x_.resize(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      A_[i][j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }

  return true;
}

bool korablev_v_jacobi_method_seq::JacobiMethodSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    std::cerr << "Error: Invalid number of inputs or outputs." << std::endl;
    return false;
  }

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (n <= 0) {
    std::cerr << "Error: Matrix size must be positive." << std::endl;
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    double diag = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
      }
    }

    if (diag <= sum) {
      std::cerr << "Error: Matrix is not diagonally dominant." << std::endl;
      return false;
    }

    if (diag == 0.0) {
      std::cerr << "Error: Zero element on the diagonal." << std::endl;
      return false;
    }
  }

  return true;
}

bool korablev_v_jacobi_method_seq::JacobiMethodSequential::run() {
  internal_order_test();
  int n = b_.size();
  std::vector<double> dx(n, 0.0);
  std::vector<double> y(n, 0.0);
  double sum;

  for (int iter = 0; iter < maxIterations_; ++iter) {
    sum = 0.0;

    for (int i = 0; i < n; ++i) {
      dx[i] = b_[i];
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          dx[i] -= A_[i][j] * x_[j];
        }
      }
      dx[i] /= A_[i][i];
      y[i] = dx[i];
      sum += std::fabs(dx[i]);
    }

    x_ = y;

    if (sum <= epsilon_) {
      break;
    }
  }

  return true;
}

bool korablev_v_jacobi_method_seq::JacobiMethodSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}