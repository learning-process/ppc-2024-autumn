// Copyright 2024 Nesterov Alexander
#include "seq/malyshev_a_simple_iteration_method/include/ops_seq.hpp"

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  uint32_t n = taskData->inputs_count[0];

  auto* ptr = reinterpret_cast<double*>(taskData->inputs[2]);
  eps_ = *reinterpret_cast<double*>(taskData->inputs[3]);
  X0_.assign(ptr, ptr + n);
  X_.resize(n);
  D_.resize(n);

  for (uint32_t i = 0; i < n; i++) D_[i] = B_[i] / A_[i][i];

  return true;
}

bool malyshev_a_simple_iteration_method_seq::TestTaskSequential::validation() {
  internal_order_test();

  // correctness of the input data
  if (taskData->inputs_count[0] != taskData->outputs_count[0] || taskData->inputs.size() != 4 ||
      taskData->outputs.empty())
    return false;

  // compatibility of a system of linear equations
  uint32_t n = taskData->inputs_count[0];
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.resize(n);
  for (uint32_t i = 0; i < n; i++, ptr += n) A_[i].assign(ptr, ptr + n);

  if (std::abs(determinant(A_)) <= std::numeric_limits<double>::epsilon()) return false;
  ptr = reinterpret_cast<double*>(taskData->inputs[1]);

  B_.assign(ptr, ptr + n);
  auto extended_matrix(A_);
  for (uint32_t i = 0; i < extended_matrix.size(); i++) {
    extended_matrix[i].push_back(B_[i]);
  }
  if (rank(A_) != rank(extended_matrix)) return false;

  // convergence of the iterative process
  C_.resize(n, std::vector<double>(n));
  for (uint32_t i = 0; i < A_.size(); i++) {
    if (A_[i][i] == 0) {
      for (uint32_t j = 0; j < A_[i].size(); j++) {
        if (i != j && A_[j][i] != 0) {
          A_[i][i] = 1;
          for (uint32_t k = 0; k < A_[i].size(); k++) {
            if (k != i) A_[i][k] = (A_[i][k] + A_[j][k]) / A_[j][i];
          }

          B_[i] = (B_[i] + B_[j]) / A_[j][i];
          break;
        }
      }
    }

    for (uint32_t j = 0; j < A_[i].size(); j++) {
      if (i == j)
        C_[i][j] = 0;
      else
        C_[i][j] = -A_[i][j] / A_[i][i];
    }
  }

  double col_sum;
  double row_sum;
  double max_col_sum = 0;
  double max_row_sum = 0;
  for (uint32_t i = 0; i < C_.size(); i++) {
    col_sum = 0;
    row_sum = 0;
    for (uint32_t j = 0; j < C_[i].size(); j++) {
      row_sum += C_[i][j];
      col_sum += C_[j][i];
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
      for (uint32_t j = 0; j < C_[i].size(); j++) {
        tmp += C_[i][j] * X0_[j];
      }
      X_[i] = tmp + D_[i];
    }

    stop = true;
    for (uint32_t i = 0; i < X_.size(); i++) {
      tmp = 0;
      for (uint32_t j = 0; j < X_.size(); j++) {
        tmp += X_[j] * A_[i][j];
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
