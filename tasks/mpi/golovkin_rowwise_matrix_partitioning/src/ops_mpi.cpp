// Golovkin Maksims

#include "mpi/golovkin_rowwise_matrix_partitioning/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::validation() {
  internal_order_test();

  rows_A = taskData->inputs_count[0];
  cols_A = taskData->inputs_count[1];
  rows_B = taskData->inputs_count[2];
  cols_B = taskData->inputs_count[3];
  return (cols_A == rows_B && rows_A > 0 && cols_A > 0 && rows_B > 0 && cols_B > 0);
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::pre_processing() {
  internal_order_test();

  A = new double[rows_A * cols_A];
  B = new double[rows_B * cols_B];
  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_ptr_a, tmp_ptr_a + rows_A * cols_A, A);
  std::copy(tmp_ptr_b, tmp_ptr_b + rows_B * cols_B, B);

  result = nullptr;

  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::run() {
  internal_order_test();

  multiply_matrices();

  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::post_processing() {
  internal_order_test();
  gather_result();
  return true;
}

void golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::multiply_matrices() {
  result = new double[rows_A * cols_B]();
  result = new double[rows_A * cols_B];
  std::fill(result, result + rows_A * cols_B, 0.0);
  for (int i = 0; i < rows_A; i++) {
    for (int j = 0; j < cols_B; j++) {
      for (int k = 0; k < cols_A; k++) {
        result[i * cols_B + j] += A[i * cols_A + k] * B[k * cols_B + j];
      }
    }
  }
}

void golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::gather_result() {
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (int i = 0; i < rows_A * cols_B; i++) {
    tmp_ptr[i] = result[i];
  }
}