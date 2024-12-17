// Copyright 2024 Sdobnov Vladimir
#include "seq/Sdobnov_V_iteration_method_yakoby/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

std::vector<double> Sdobnov_iteration_method_yakoby_seq::iteration_method_yakoby(int n, const std::vector<double>& A,
                                                                                 const std::vector<double>& b,
                                                                                 double tolerance, int maxIterations) {
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i * n + i];
    }
    double max_diff = 0.0;

    for (int i = 0; i < n; ++i) {
      max_diff = std::max(max_diff, fabs(x_new[i] - x[i]));
    }

    if (max_diff < tolerance) {
      break;
    }

    x = x_new;
  }

  return x_new;
}

static bool isDiagonallyDominant(int n, const std::vector<double>& A) {
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        sum += fabs(A[i * n + j]);
      }
    }

    if (fabs(A[i * n + i]) <= sum) {
      return false;
    }

    if (fabs(A[i * n + i]) < 1e-9) {
      return false;
    }
  }

  return true;
}

bool Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq::pre_processing() {
  internal_order_test();

  size_ = taskData->inputs_count[0];
  matrix_.assign(size_ * size_, 0.0);
  free_members_.assign(size_, 0.0);
  res_.assign(size_, 0.0);

  auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* pfree_members = reinterpret_cast<double*>(taskData->inputs[1]);

  std::copy(pmatrix, pmatrix + size_ * size_, matrix_.begin());
  std::copy(pfree_members, pfree_members + size_, free_members_.begin());

  return true;
}

bool Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq::validation() {
  internal_order_test();

  bool correct_count = taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 &&
                       taskData->inputs.size() == 2 && taskData->outputs_count.size() == 1 &&
                       taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1;
  if (!correct_count) return false;

  int size = taskData->inputs_count[0];
  std::vector<double> matrix(size * size, 0.0);
  auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(pmatrix, pmatrix + size * size, matrix.begin());

  return isDiagonallyDominant(size, matrix);
}

bool Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq::run() {
  internal_order_test();
  res_ = iteration_method_yakoby(size_, matrix_, free_members_, tolerance, maxIterations);
  return true;
}

bool Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}
