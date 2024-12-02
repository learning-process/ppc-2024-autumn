// Copyright 2024 Nesterov Alexander
#include "seq/malyshev_a_simple_iteration_method/include/ops_seq.hpp"

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<double*>(taskData->inputs[2]);
  eps_ = *reinterpret_cast<double*>(taskData->inputs[3]);
  X0_.assign(ptr, ptr + n_);
  X_.resize(n_);
  D_.resize(n_);

  for (uint32_t i = 0; i < n_; i++) D_[i] = B_[i] / A_[i * n_ + i];

  return true;
}

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  // correctness of the input data
  if (taskData->inputs_count.empty() || taskData->outputs_count.empty() ||
      taskData->inputs_count[0] != taskData->outputs_count[0] || taskData->inputs.size() != 4 ||
      taskData->outputs.empty())
    return false;

  // compatibility of a system of linear equations
  n_ = taskData->inputs_count[0];
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(ptr, ptr + n_ * n_);

  if (std::abs(determinant(A_, n_)) <= std::numeric_limits<double>::epsilon()) return false;
  ptr = reinterpret_cast<double*>(taskData->inputs[1]);

  B_.assign(ptr, ptr + n_);
  uint32_t m = n_ + 1;
  std::vector<double> extended_matrix(m * n_);
  for (uint32_t i = 0; i < n_; i++) {
    for (uint32_t j = 0; j < m; j++) {
      if (j == m - 1) {
        extended_matrix[i * m + j] = B_[i];
      } else {
        extended_matrix[i * m + j] = A_[i * n_ + j];
      }
    }
  }
  if (rank(A_, n_, n_) != rank(extended_matrix, n_, m)) return false;

  // convergence of the iterative process
  C_.resize(n_ * n_);
  for (uint32_t i = 0; i < n_; i++) {
    if (A_[i * n_ + i] == 0) {
      for (uint32_t j = 0; j < n_; j++) {
        if (i != j && A_[j * n_ + i] != 0) {
          A_[i * n_ + i] = 1;
          for (uint32_t k = 0; k < n_; k++) {
            if (k != i) A_[i * n_ + k] = (A_[i * n_ + k] + A_[j * n_ + k]) / A_[j * n_ + i];
          }

          B_[i] = (B_[i] + B_[j]) / A_[j * n_ + i];
          break;
        }
      }
    }

    for (uint32_t j = 0; j < n_; j++) {
      if (i == j)
        C_[i * n_ + j] = 0;
      else
        C_[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
    }
  }

  double col_sum;
  double row_sum;
  double max_col_sum = 0;
  double max_row_sum = 0;
  for (uint32_t i = 0; i < n_; i++) {
    col_sum = 0;
    row_sum = 0;
    for (uint32_t j = 0; j < n_; j++) {
      row_sum += C_[i * n_ + j];
      col_sum += C_[j * n_ + i];
    }

    max_col_sum = std::max(max_col_sum, std::abs(col_sum));
    max_row_sum = std::max(max_row_sum, std::abs(row_sum));
  }

  return max_col_sum <= 1 || max_row_sum <= 1;
}

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::run() {
  internal_order_test();

  double tmp;
  bool stop = false;
  while (!stop) {
    for (uint32_t i = 0; i < X_.size(); i++) {
      tmp = 0;
      for (uint32_t j = 0; j < n_; j++) {
        tmp += C_[i * n_ + j] * X0_[j];
      }
      X_[i] = tmp + D_[i];
    }

    stop = true;
    for (uint32_t i = 0; i < X_.size(); i++) {
      tmp = 0;
      for (uint32_t j = 0; j < X_.size(); j++) {
        tmp += X_[j] * A_[i * n_ + j];
      }
      if (std::abs(tmp - B_[i]) > eps_) {
        stop = false;
        break;
      }
    }

    X0_ = X_;
  }

  return true;
}

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(X_.begin(), X_.end(), out);

  return true;
}
