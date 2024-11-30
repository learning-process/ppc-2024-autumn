// Copyright 2024 Nesterov Alexander
#include "seq/rezantseva_a_simple_iteration_method/include/ops_seq_rezantseva.hpp"

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::isTimeToStop(
    const std::vector<double>& x0, const std::vector<double>& x1) const {
  double max_precision = 0.0;  // max precision between iterations

  for (size_t k = 0; k < x0.size(); k++) {
    double precision = std::fabs(x1[k] - x0[k]);  // |x1^(i+1) - x1^i|
    if (precision > max_precision) {
      max_precision = precision;
    }
  }
  return (max_precision < epsilon_);
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::checkMatrix() {
  for (size_t i = 0; i < n_; ++i) {  // row

    double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n_; ++j) {  // column
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j]);
      }
    }
    if (Aii <= sum) {
      return false;
    }
  }
  return true;
}
bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::checkMatrixNorm() {
  double max_row_sum = 0.0;
  for (size_t i = 0; i < n_; ++i) {
    double row_sum = 0.0;
    for (size_t j = 0; j < n_; ++j) {
      row_sum += std::abs(B_[i * n_ + j]);
    }
    max_row_sum = std::max(max_row_sum, row_sum);
  }
  return max_row_sum < 1.0;
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::validation() {
  internal_order_test();
  n_ = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n_ > 0) && checkMatrix();
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::pre_processing() {
  internal_order_test();
  A_.assign(n_ * n_, 0.0);
  b_.assign(n_, 0.0);
  x_.assign(n_, 0.0);

  B_.assign(n_ * n_, 0.0);
  c_.assign(n_, 0.0);
  // fill matrix A and vector b
  for (size_t i = 0; i < n_; ++i) {    // row
    for (size_t j = 0; j < n_; ++j) {  // column
      A_[i * n_ + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }
  // fill transition matrix B and iteration vector c
  for (size_t i = 0; i < n_; ++i) {
    double diag = A_[i * n_ + i];
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {                               // diagonal elements of B remain zero
        B_[i * n_ + j] = -A_[i * n_ + j] / diag;  // Bij = -Aij/Aii
      }
    }
    c_[i] = b_[i] / diag;  // ci = bi/Aii
  }

  return checkMatrixNorm();
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::run() {
  internal_order_test();
  size_t iteration = 0;
  std::vector<double> prev_x(n_, 0.0);

  while (iteration < maxIteration_) {
    std::copy(x_.begin(), x_.end(), prev_x.begin());  // saved previous approach

    // new approach x = Bx + c
    for (size_t i = 0; i < n_; i++) {
      x_[i] = c_[i];
      for (size_t j = 0; j < n_; j++) {
        x_[i] += B_[i * n_ + j] * prev_x[j];
      }
    }
    if (isTimeToStop(prev_x, x_)) {
      break;
    }
    iteration++;
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}
