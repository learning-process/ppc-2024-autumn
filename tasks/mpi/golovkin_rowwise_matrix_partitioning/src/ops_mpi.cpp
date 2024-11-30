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
  rows_A = taskData->inputs_count[0];
  cols_A = taskData->inputs_count[1];
  rows_B = taskData->inputs_count[2];
  cols_B = taskData->inputs_count[3];

  bool local_valid = (cols_A == rows_B && rows_A > 0 && cols_A > 0 && rows_B > 0 && cols_B > 0);
  bool global_valid = boost::mpi::all_reduce(world, local_valid, std::logical_and<bool>());
  return global_valid;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::pre_processing() {
  if (world.size() < 5 || world.rank() >= 4) {
    auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);

    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];

    std::copy(tmp_ptr_a, tmp_ptr_a + rows_A * cols_A, A);
    std::copy(tmp_ptr_b, tmp_ptr_b + rows_B * cols_B, B);
  }

  distribute_matrices();
  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::run() {
  perform_local_multiplication();
  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::post_processing() {
  gather_result();
  return true;
}

void golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::distribute_matrices() {
  if (world.size() < 5 || world.rank() >= 4) {
    boost::mpi::broadcast(world, A, rows_A * cols_A, 0);
    boost::mpi::broadcast(world, B, rows_B * cols_B, 0);
  } else {
    boost::mpi::broadcast(world, A, rows_A * cols_A, 0);
    boost::mpi::broadcast(world, B, rows_B * cols_B, 0);
  }
}

void golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::perform_local_multiplication() {
  // Определяем диапазон строк для текущего процесса
  int rows_per_rank = rows_A / world.size();
  int extra_rows = rows_A % world.size();
  int start_row = world.rank() * rows_per_rank + std::min(world.rank(), extra_rows);
  int end_row = start_row + rows_per_rank + (world.rank() < extra_rows ? 1 : 0);

  int local_rows = end_row - start_row;

  // Вычисляем локальную часть матрицы C
  local_result = new double[local_rows * cols_B]();
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < cols_B; j++) {
      for (int k = 0; k < cols_A; k++) {
        local_result[i * cols_B + j] += A[(start_row + i) * cols_A + k] * B[k * cols_B + j];
      }
    }
  }
  local_rows_count = local_rows;  // Сохраняем количество локальных строк для дальнейшего использования
}

void golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::gather_result() {
  if (world.rank() == 0) {
    result = new double[rows_A * cols_B]();
  }

  // Сбор результатов со всех процессов
  std::vector<int> recvcounts(world.size());
  std::vector<int> displs(world.size());

  if (world.size() < 5 || world.rank() >= 4) {
    int offset = 0;
    for (int i = 0; i < world.size(); i++) {
      int rows_for_rank = rows_A / world.size() + (i < rows_A % world.size() ? 1 : 0);
      recvcounts[i] = rows_for_rank * cols_B;
      displs[i] = offset;
      offset += recvcounts[i];
    }
  }

  boost::mpi::gatherv(world, local_result, local_rows_count * cols_B, result, recvcounts, displs, 0);

  // Только процесс 0 сохраняет финальный результат
  if (world.size() < 5 || world.rank() >= 4) {
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(result, result + rows_A * cols_B, tmp_ptr);
  }

  // Очистка памяти
  delete[] local_result;
}
