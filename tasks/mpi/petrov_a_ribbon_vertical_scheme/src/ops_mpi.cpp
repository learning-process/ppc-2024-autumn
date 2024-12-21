// Copyright 2024 Nesterov Alexander
#include "mpi/petrov_a_ribbon_vertical_scheme/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

using namespace std::chrono_literals;

bool petrov_a_ribbon_vertical_scheme_mpi::TestTaskMPI::pre_processing() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    int rows = taskData->inputs_count[0];
    int cols = taskData->inputs_count[1];

    int rows_per_process = rows / size;
    int extra_rows = rows % size;

    int start_row = rank * rows_per_process + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    local_matrix.resize((end_row - start_row) * cols);
    local_vector.resize(cols);
    local_result.resize(end_row - start_row);
  }

  return true;
}

bool petrov_a_ribbon_vertical_scheme_mpi::TestTaskMPI::validation() {
  internal_order_test();
  bool isValid = true;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    isValid = (!taskData->inputs_count.empty() && !taskData->inputs.empty() && !taskData->outputs.empty());
  }
  return isValid;
}

bool petrov_a_ribbon_vertical_scheme_mpi::TestTaskMPI::run() {
  internal_order_test();
  int size;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int cols = 0;
  if (rank == 0) {
    cols = taskData->inputs_count[1];
  }

  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rows = taskData->inputs_count[0];
  int rows_per_proc = rows / size;

  local_matrix.resize(rows_per_proc * cols);
  local_vector.resize(cols);

  MPI_Scatter(taskData->inputs.data(), rows_per_proc * cols, MPI_DOUBLE, local_matrix.data(), rows_per_proc * cols,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(local_vector.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < rows_per_proc; ++i) {
    for (int j = 0; j < cols; ++j) {
      local_matrix[i * cols + j] *= local_vector[j];
    }
  }

  return true;
}

bool petrov_a_ribbon_vertical_scheme_mpi::TestTaskMPI::post_processing() {
  internal_order_test();
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  int rows_per_proc = rows / size;

  std::vector<double> global_result(rows * cols);

  MPI_Gather(local_matrix.data(), rows_per_proc * cols, MPI_DOUBLE, global_result.data(), rows_per_proc * cols,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}
