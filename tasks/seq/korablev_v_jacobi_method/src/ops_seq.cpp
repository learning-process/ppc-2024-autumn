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
    std::cerr << "Incorrect number of input or output data" << std::endl;
    return false;
  }

  return true;
}

bool korablev_v_jacobi_method_seq::JacobiMethodSequential::run() {
  internal_order_test();
  int n = b_.size();
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < maxIterations_; ++iter) {
    for (int i = 0; i < n; ++i) {
      x_new[i] = b_[i];

      for (int j = 0; j < n; ++j) {
        if (j != i) {
          x_new[i] -= A_[i][j] * x_[j];
        }
      }

      x_new[i] /= A_[i][i];
    }

    double maxDiff = 0.0;
    for (int i = 0; i < n; ++i) {
      maxDiff = std::max(maxDiff, std::fabs(x_new[i] - x_[i]));
    }

    x_ = x_new;

    if (maxDiff < epsilon_) {
      std::cout << "Сходимость достигнута после " << iter << " итераций." << std::endl;
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
