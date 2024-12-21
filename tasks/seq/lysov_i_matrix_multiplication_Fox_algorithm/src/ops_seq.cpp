// Copyright 2024 Nesterov Alexander
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  N = reinterpret_cast<std::size_t*>(taskData->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t*>(taskData->inputs[3])[0];
  A.resize(N * N);
  B.resize(N * N);
  C.resize(N * N, 0.0);
  std::copy(reinterpret_cast<double*>(taskData->inputs[1]), reinterpret_cast<double*>(taskData->inputs[1]) + N * N,
            A.begin());
  std::copy(reinterpret_cast<double*>(taskData->inputs[2]), reinterpret_cast<double*>(taskData->inputs[2]) + N * N,
            B.begin());
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::validation() {
  internal_order_test();
  N = reinterpret_cast<std::size_t*>(taskData->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t*>(taskData->inputs[3])[0];
  if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
    return false;
  }
  if (taskData->inputs_count[1] != N * N || taskData->inputs_count[0] != N * N) {
    return false;
  }
  return taskData->outputs_count[0] == N * N && block_size > 0;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::run() {
  internal_order_test();
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        double a_ij = A[i * N + k];
        double b_kj = B[k * N + j];
        sum += a_ij * b_kj;
      }
      C[i * N + j] = sum;
    }
  }
  return true;
}
bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(C.begin(), C.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}