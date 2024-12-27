// Copyright 2024 Nesterov Alexander
#include "seq/frolova_e_matrix_multiplication/include/ops_seq_frolova_matrix.hpp"

#include <thread>

using namespace std::chrono_literals;

std::vector<int> frolova_e_matrix_multiplication_seq::Multiplication(size_t M, size_t N, size_t K,
                                                                     const std::vector<int>& A,
                                                                     const std::vector<int>& B) {
  std::vector<int> C(M * N);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
  return C;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::pre_processing() {
  internal_order_test();

  // Init value for input and output
  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  lineA = static_cast<size_t>(value_1[0]);
  columnA = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  lineB = static_cast<size_t>(value_2[0]);
  columnB = static_cast<size_t>(value_2[1]);

  int* matr1_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
  matrixA.assign(matr1_ptr, matr1_ptr + taskData->inputs_count[2]);

  int* matr2_ptr = reinterpret_cast<int*>(taskData->inputs[3]);
  matrixB.assign(matr2_ptr, matr2_ptr + taskData->inputs_count[3]);

  matrixC.resize(lineA * columnB);

  return true;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::validation() {
  internal_order_test();

  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  if (taskData->inputs_count[0] != 2) {
    return false;
  }

  auto line1 = static_cast<size_t>(value_1[0]);
  auto column1 = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  if (taskData->inputs_count[1] != 2) {
    return false;
  }

  auto line2 = static_cast<size_t>(value_2[0]);
  auto column2 = static_cast<size_t>(value_2[1]);

  if (value_1[1] != value_2[0]) {
    return false;
  }

  if (taskData->inputs_count[2] != line1 * column1) {
    return false;
  }

  if (taskData->inputs_count[3] != line2 * column2) {
    return false;
  }

  if (taskData->outputs_count[0] != line1 * column2) {
    return false;
  }

  return true;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::run() {
  internal_order_test();
  matrixC = Multiplication(lineA, columnB, columnA, matrixA, matrixB);

  return true;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::post_processing() {
  internal_order_test();

  for (size_t i = 0; i < lineA * columnB; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = matrixC[i];
  }

  return true;
}