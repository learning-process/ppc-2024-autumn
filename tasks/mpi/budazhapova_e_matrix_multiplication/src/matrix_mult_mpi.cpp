#include "mpi/budazhapova_e_matrix_multiplication/include/matrix_mult_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

bool budazhapova_e_matrix_mult_mpi::MatrixMultSequential::pre_processing() {
  internal_order_test();
  A = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                       reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  b = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[1]),
                       reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1]);
  columns = taskData->inputs_count[1];
  rows = taskData->inputs_count[0] / columns;
  res = std::vector<int>(rows);
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] % columns == 0 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows; i++) {
    res[i] = 0;
    for (int j = 0; j < columns; j++) {
      res[i] += A[j + columns * i] * b[j];
    }
  }
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultSequential::post_processing() {
  internal_order_test();
  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  for (int i = 0; i < rows; i++) {
    output[i] = res[i];
  }

  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::pre_processing() {
  internal_order_test();
  int world_rank = world.rank();
  int world_size = world.size();

  if (world_rank == 0) {
    A = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                         reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    b = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[1]),
                         reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1]);
    columns = taskData->inputs_count[1];
    rows = taskData->inputs_count[0] / columns;
    res = std::vector<int>(rows);

    boost::mpi::broadcast(world, A, 0);
    boost::mpi::broadcast(world, b, 0);
  }

  if (rows >= world_size) {
    int n_of_send_rows = rows / world_size;
    int n_of_proc_with_extra_row = rows % world_size;

    int start_row = world_rank * n_of_send_rows + std::min(world_rank, n_of_proc_with_extra_row);
    int end_row = start_row + n_of_send_rows + (world_rank < n_of_proc_with_extra_row ? 1 : 0);
    local_res.resize(end_row - start_row, 0);
  } else {
    for (int i = 0; i < rows; ++i) {
      res[i] = 0;
      for (int j = 0; j < columns; ++j) {
        res[i] += A[i * columns + j] * b[j];
      }
    }
  }
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs_count[0] % columns == 0 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::run() {
  internal_order_test();

  int n_of_send_rows = rows / world.size();
  int n_of_proc_with_extra_row = rows % world.size();

  int start_row = world.rank() * n_of_send_rows + std::min(world.rank(), n_of_proc_with_extra_row);
  int end_row = start_row + n_of_send_rows + (world.rank() < n_of_proc_with_extra_row ? 1 : 0);

  for (int i = start_row; i < end_row; i++) {
    local_res[i - start_row] = 0;
    for (int j = 0; j < columns; j++) {
      local_res[i - start_row] += A[j + columns * i] * b[j];
    }
  }
  boost::mpi::gather(world, local_res.data(), local_res.size(), res.data(), 0);
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output = reinterpret_cast<int*>(taskData->outputs[0]);
    for (int i = 0; i < rows; i++) {
      output[i] = res[i];
    }
  }
  return true;
}
