// Copyright 2024 Nesterov Alexander
#include "seq/lysov_i_simple_iteration_method/include/ops_seq.hpp"

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::isDiagonallyDominant() {
  for (int i = 0; i < input_size_; ++i) {
    double diagonal = std::abs(A_[i][i]);
    double sum = 0.0;

    for (int j = 0; j < input_size_; ++j) {
      if (i != j) {
        sum += std::abs(A_[i][j]);
      }
    }

    if (diagonal <= sum) {
      return false;
    }
  }

  return true;
}

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::transformSystem() {
  B_.resize(input_size_, std::vector<double>(input_size_, 0.0));
  g_.resize(input_size_);

  for (int i = 0; i < input_size_; ++i) {
    double diagonal = A_[i][i];

    if (diagonal == 0.0) {
      std::cerr << "Zero diagonal element detected, cannot apply iteration.";
      return false;
    }

    g_[i] = b_[i] / diagonal;

    for (int j = 0; j < input_size_; ++j) {
      if (i != j) {
        B_[i][j] = -A_[i][j] / diagonal;
      }
    }
  }
  return true;
}

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::pre_processing() {
  internal_order_test();

  auto* X_raw = reinterpret_cast<double*>(taskData->outputs[0]);
  x_.resize(input_size_);
  for (int i = 0; i < input_size_; ++i) {
    x_[i] = X_raw[i];
  }

  tolerance_ = 1e-6;

  return true;
}

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] != taskData->inputs_count[1] || taskData->inputs_count[0] != taskData->outputs_count[0])
    return false;
  input_size_ = taskData->inputs_count[0];
  A_.resize(input_size_, std::vector<double>(input_size_));
  auto* A_raw = reinterpret_cast<double*>(taskData->inputs[0]);

  for (int i = 0; i < input_size_; ++i) {
    for (int j = 0; j < input_size_; ++j) {
      A_[i][j] = A_raw[i * input_size_ + j];
    }
  }

  auto* B_raw = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.resize(input_size_);
  for (int i = 0; i < input_size_; ++i) {
    b_[i] = B_raw[i];
  }

  if (!isDiagonallyDominant()) return false;
  if (!transformSystem()) return false;
  return true;
}

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::run() {
  internal_order_test();

  std::vector<double> x_new(input_size_, 0.0);
  double max_diff = 0.0;

  do {
    max_diff = 0.0;

    for (int i = 0; i < input_size_; ++i) {
      double sum = 0.0;

      for (int j = 0; j < input_size_; ++j) {
        if (i != j) {
          sum += B_[i][j] * x_[j];
        }
      }

      x_new[i] = g_[i] + sum;

      double diff = std::abs(x_new[i] - x_[i]);
      max_diff = std::max(max_diff, diff);
    }

    x_ = x_new;

  } while (max_diff > tolerance_);

  return true;
}

bool lysov_i_simple_iteration_method_seq::SlaeIterationTask::post_processing() {
  internal_order_test();
  for (int i = 0; i < static_cast<int>(x_.size()); i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}