// Copyright 2024 Nesterov Alexander
#include "seq/veliev_e_jacobi_method/include/ops_seq.hpp"

bool veliev_e_jacobi_method::MethodJacobi::pre_processing() {
  internal_order_test();

  auto* matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* rhs = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* initial_guess = reinterpret_cast<double*>(taskData->inputs[2]);
  N = static_cast<int>(taskData->inputs_count[0]);
  matrixA.resize(N * N);
  rshB.resize(N);
  initialGuessX.resize(N);
  eps = 1e-9;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrixA[i * N + j] = matrix[i * N + j];
    }
    rshB[i] = rhs[i];
    initialGuessX[i] = initial_guess[i];
  }

  for (int i = 0; i < N; i++) {
    if (matrixA[i * N + i] == 0) {
      std::cerr << "Incorrect matrix: diagonal element matrixA[" << i + 1 << "][" << i + 1 << "] is zero.";
      return false;
    }
  }

  return true;
}

bool veliev_e_jacobi_method::MethodJacobi::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

void veliev_e_jacobi_method::MethodJacobi::iteration_J() {
  std::vector<double> TempX(N);

  for (int i = 0; i < N; i++) {
    TempX[i] = rshB[i];
    for (int j = 0; j < N; j++) {
      if (i != j) TempX[i] -= matrixA[i * N + j] * initialGuessX[j];
    }
    TempX[i] /= matrixA[i * N + i];
  }

  for (int h = 0; h < N; h++) {
    initialGuessX[h] = TempX[h];
  }
}

bool veliev_e_jacobi_method::MethodJacobi::run() {
  internal_order_test();
  double norm;
  std::vector<double> prev_X(N);
  const int max_iterations = 1000000;
  int iteration_count = 0;
  do {
    prev_X = initialGuessX;

    iteration_J();

    norm = fabs(initialGuessX[0] - prev_X[0]);
    for (int i = 0; i < N; i++) {
      if (fabs(initialGuessX[i] - prev_X[i]) > norm) norm = fabs(initialGuessX[i] - prev_X[i]);
    }
    iteration_count++;
  } while (norm > eps && iteration_count < max_iterations);
  return true;
}

bool veliev_e_jacobi_method::MethodJacobi::post_processing() {
  internal_order_test();
  for (int i = 0; i < N; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = initialGuessX[i];
  }
  return true;
}
