// Copyright 2024 Nesterov Alexander
#include "seq/kozlova_e_jacobi_method/include/ops_seq.hpp"

bool kozlova_e_jacobi_method::MethodJacobi::pre_processing() {
  internal_order_test();

  auto* matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* initial_guess = reinterpret_cast<double*>(taskData->inputs[2]);
  N = static_cast<int>(taskData->inputs_count[0]);
  A.resize(N * N);
  B.resize(N);
  X.resize(N);
  eps = 1e-9;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = matrix[i * N + j];
    }
    B[i] = rhs[i];
    X[i] = initial_guess[i];
  }

  for (int i = 0; i < N; i++) {
    if (A[i * N + i] == 0) {
      std::cerr << "Incorrect matrix: diagonal element A[" << i + 1 << "][" << i + 1 << "] is zero.";
      return false;
    }
  }

  return true;
}

bool kozlova_e_jacobi_method::MethodJacobi::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

void kozlova_e_jacobi_method::MethodJacobi::jacobi_iteration() {
  std::vector<double> TempX(N);

  for (int i = 0; i < N; i++) {
    TempX[i] = B[i];
    for (int j = 0; j < N; j++) {
      if (i != j) TempX[i] -= A[i * N + j] * X[j];
    }
    TempX[i] /= A[i * N + i];
  }

  for (int h = 0; h < N; h++) {
    X[h] = TempX[h];
  }
}

bool kozlova_e_jacobi_method::MethodJacobi::run() {
  internal_order_test();
  double norm;
  std::vector<double> prev_X(N);
  const int max_iterations = 1000000;
  int iteration_count = 0;
  do {
    prev_X = X;

    jacobi_iteration();

    norm = fabs(X[0] - prev_X[0]);
    for (int i = 0; i < N; i++) {
      if (fabs(X[i] - prev_X[i]) > norm) norm = fabs(X[i] - prev_X[i]);
    }
    iteration_count++;
  } while (norm > eps && iteration_count < max_iterations);
  return true;
}

bool kozlova_e_jacobi_method::MethodJacobi::post_processing() {
  internal_order_test();
  for (int i = 0; i < N; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = X[i];
  }
  return true;
}
