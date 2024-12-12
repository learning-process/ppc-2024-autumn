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
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
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

  std::vector<int> recv_counts(world.size(), 0);
  std::vector<int> displacements(world.size(), 0);

  if (world.rank() == 0) {
    A = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                         reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    b = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[1]),
                         reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1]);
    columns = taskData->inputs_count[1];
    rows = taskData->inputs_count[0] / columns;
    res = std::vector<int>(rows, 0);
  }
  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
           taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  }

  return true;
}

bool budazhapova_e_matrix_mult_mpi::MatrixMultParallel::run() {
  internal_order_test();
  std::vector<int> recv_counts(world.size(), 0);
  std::vector<int> displacements(world.size(), 0);

  boost::mpi::broadcast(world, columns, 0);
  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, A, 0);
  boost::mpi::broadcast(world, b, 0);

  int n_of_send_rows;
  int n_of_proc_with_extra_row;
  int start_row;
  int end_row;
  int world_size = world.size();
  int world_rank = world.rank();

  for (int i = 0; i < world_size; i++) {
    n_of_send_rows = rows / world_size;
    n_of_proc_with_extra_row = rows % world_size;

    start_row = i * n_of_send_rows + std::min(i, n_of_proc_with_extra_row);
    end_row = start_row + n_of_send_rows + (i < n_of_proc_with_extra_row ? 1 : 0);
    recv_counts[i] = end_row - start_row;
    displacements[i] = (i == 0) ? 0 : displacements[i - 1] + recv_counts[i - 1];
  }

  n_of_send_rows = rows / world_size;
  n_of_proc_with_extra_row = rows % world_size;

  start_row = world_rank * n_of_send_rows + std::min(world_rank, n_of_proc_with_extra_row);
  end_row = start_row + n_of_send_rows + (world_rank < n_of_proc_with_extra_row ? 1 : 0);

  if (world.size() > rows) {
    if (world.rank() < rows) {
      local_A.resize(columns);
      local_res.resize(1);
      for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < columns; j++) {
          local_A[(i - start_row) * columns + j] = A[i * columns + j];
        }
      }
    } else {
      local_A.clear();
      local_res.clear();
      return true;
    }
  } else {
    local_A.resize((end_row - start_row) * columns);
    local_res.resize(end_row - start_row, 0);

    for (int i = start_row; i < end_row; i++) {
      for (int j = 0; j < columns; j++) {
        local_A[(i - start_row) * columns + j] = A[i * columns + j];
      }
    }
  }

  for (size_t i = 0; i < local_res.size(); i++) {
    local_res[i] = 0;
    for (int j = 0; j < columns; j++) {
      local_res[i] += local_A[i * columns + j] * b[j];
    }
  }
  res.resize(rows);
  boost::mpi::gatherv(world, local_res.data(), local_res.size(), res.data(), recv_counts, displacements, 0);
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
