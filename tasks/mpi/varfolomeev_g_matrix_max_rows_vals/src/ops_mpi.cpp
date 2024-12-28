#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <vector>

// Sequential
bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_m = taskData->inputs_count[0];  // rows count
  size_n = taskData->inputs_count[1];  // columns count
  res_vec = std::vector<int>(size_m, 0);
  mtr = std::vector<int>(size_m * size_n, 0);  // init matrix (row-alike matrix)

  // Заполняем mtr данными из taskData
  auto *inpt_prt = reinterpret_cast<int *>(taskData->inputs[0]);
  for (int i = 0; i < size_m * size_n; i++) {
    mtr[i] = inpt_prt[i];
  }

  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 2 &&   // Checking that there are two elements in inputs_count
         taskData->inputs_count[0] >= 0 &&       // Checking that the number of rows is greater than 0
         taskData->inputs_count[1] >= 0 &&       // Checking that the number of columns is greater than 0
         taskData->outputs_count.size() == 1 &&  // Checking that there is one element in outputs_count
         taskData->outputs_count[0] ==
             taskData->inputs_count[0];  // Checking that the number of output data is equal to the number of rows
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::run() {
  internal_order_test();
  for (int i = 0; i < size_m; i++) {
    int row_start = i * size_n;        // beginning of the curr line
    int row_end = row_start + size_n;  // end of the curr line
    res_vec[i] = *std::max_element(mtr.begin() + row_start, mtr.begin() + row_end);
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_m; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = res_vec[i];
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::pre_processing() {
  internal_order_test();
  // Init vectors
  size_m = taskData->inputs_count[0];
  size_n = taskData->inputs_count[1];

  if (world.rank() == 0) {
    mtr = std::vector<int>(size_n * size_m);
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    mtr = std::vector<int>(tmp_ptr, tmp_ptr + size_n * size_m);
    // Init values for output
    res_vec = std::vector<int>(size_m, 0);
  }

  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count.size() == 2 &&                     // dimensions of the matrix have been transmitted
           taskData->inputs_count[0] >= 0 &&                         // number of rows is non-negative
           taskData->inputs_count[1] >= 0 &&                         // number of cols is non-negative
           taskData->outputs_count.size() == 1 &&                    // there is only one output
           taskData->outputs_count[0] == taskData->inputs_count[0];  // number of outputs is equal to the number of rows
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::run() {
  internal_order_test();
  if (size_m == 0 || size_n == 0) {
    res_vec = std::vector<int>(0);  // empty res
    return true;                    // do nothing for an empty matrix
  }

  if (world.rank() == 0) {  // root process
    // Calculating the number of rows for each process
    std::vector<int> counts(world.size());
    std::vector<int> displs(world.size());
    int rows_per_process = size_m / world.size();
    int remainder = size_m % world.size();

    for (int i = 0; i < world.size(); ++i) {
      counts[i] = rows_per_process * size_n;
      if (i < remainder) {
        counts[i] += size_n;
      }
      displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    // share data between processes using MPI_Scatterv
    std::vector<int> local_data(counts[world.rank()]);
    MPI_Scatterv(mtr.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[world.rank()], MPI_INT, 0,
                 MPI_COMM_WORLD);

    // Calculating local maxima for the root process
    std::vector<int> local_maxes(local_data.size() / size_n);
    for (size_t i = 0; i < local_maxes.size(); ++i) {
      local_maxes[i] = *std::max_element(local_data.begin() + i * size_n, local_data.begin() + (i + 1) * size_n);
    }

    // Collecting the results using MPI_Gatherv
    std::vector<int> all_maxes(size_m);
    std::vector<int> recv_counts(world.size());
    std::vector<int> recv_displs(world.size());
    for (int i = 0; i < world.size(); ++i) {
      recv_counts[i] = counts[i] / size_n;
      recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
    }

    MPI_Gatherv(local_maxes.data(), local_maxes.size(), MPI_INT, all_maxes.data(), recv_counts.data(),
                recv_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    res_vec = all_maxes;
  } else {  // other processes
    // Calculating the number of rows for curr process
    int rows_per_process = size_m / world.size();
    int remainder = size_m % world.size();
    int local_rows = rows_per_process + (world.rank() < remainder ? 1 : 0);
    std::vector<int> local_data(local_rows * size_n);

    // Collecting data between processes using MPI_Scatterv
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), local_data.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // Calculating local maxima for curr process
    std::vector<int> local_maxes(local_rows);
    for (int i = 0; i < local_rows; ++i) {
      local_maxes[i] = *std::max_element(local_data.begin() + i * size_n, local_data.begin() + (i + 1) * size_n);
    }

    // Sending the results using MPI_Gatherv
    MPI_Gatherv(local_maxes.data(), local_maxes.size(), MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0 && size_m > 0 && size_n > 0) {
    for (int i = 0; i < size_m; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = res_vec[i];
    }
  }
  return true;
}