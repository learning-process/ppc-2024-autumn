#include "mpi/savchenko_m_ribbon_mult_split_a/include/ops_mpi_savchenko.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

// Task Sequential

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  // columns_A = taskData->inputs_count[0];
  // rows_A = taskData->inputs_count[1];
  // columns_B = taskData->inputs_count[2];
  // rows_B = taskData->inputs_count[3];

  bool valid_inputs = (taskData->inputs.size() == 2);
  bool valid_outputs = (taskData->outputs.size() == 1);
  bool valid_icount = (taskData->inputs_count.size() == 4);
  bool valid_ocount = (taskData->outputs_count.size() == 1);
  bool valid_io = (valid_inputs && valid_outputs && valid_icount && valid_ocount);
  if (!valid_io) return false;

  bool matrix_A_positive_size = (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0);
  bool matrix_B_positive_size = (taskData->inputs_count[2] > 0 && taskData->inputs_count[3] > 0);
  bool equal_columnsA_rowsB = (taskData->inputs_count[0] == taskData->inputs_count[3]);

  bool valid = (matrix_A_positive_size && matrix_B_positive_size && equal_columnsA_rowsB);
  return valid;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init values for input and output
  columns_A = taskData->inputs_count[0];
  rows_A = taskData->inputs_count[1];
  columns_B = taskData->inputs_count[2];
  rows_B = taskData->inputs_count[3];

  matrix_A = std::vector<int>(columns_A * rows_A, 0);
  matrix_B = std::vector<int>(columns_B * rows_B, 0);
  matrix_res = std::vector<int>(rows_A * columns_B, 0);

  auto *tmp_A = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_A, tmp_A + rows_A * columns_A, matrix_A.begin());

  auto *tmp_B = reinterpret_cast<int *>(taskData->inputs[1]);
  std::copy(tmp_B, tmp_B + rows_B * columns_B, matrix_B.begin());

  return true;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows_A; i++) {
    for (int j = 0; j < columns_B; j++) {
      for (int k = 0; k < columns_A; k++) {
        matrix_res[i * columns_B + j] += matrix_A[i * columns_A + k] * matrix_B[k * columns_B + j];
      }
    }
  }
  return true;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int *ptr_matrix_res = reinterpret_cast<int *>(taskData->outputs[0]);
  std::copy(matrix_res.begin(), matrix_res.begin() + rows_A * columns_B, ptr_matrix_res);
  return true;
}

// Task Parallel

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    bool valid_inputs = (taskData->inputs.size() == 2);
    bool valid_outputs = (taskData->outputs.size() == 1);
    bool valid_icount = (taskData->inputs_count.size() == 1);
    bool valid_ocount = (taskData->outputs_count.size() == 1);
    bool valid_io = (valid_inputs && valid_outputs && valid_icount && valid_ocount);
    if (!valid_io) return false;

    bool valid_size = (taskData->inputs_count[0] > 0);
    return valid_size;
  }
  return true;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init values for input and output
    size = taskData->inputs_count[0];

    int vect_size = size * size;

    matrix_A = std::vector<int>(vect_size, 0);
    matrix_B = std::vector<int>(vect_size, 0);
    matrix_res = std::vector<int>(vect_size, 0);

    auto *tmp_A = reinterpret_cast<int *>(taskData->inputs[0]);
    std::copy(tmp_A, tmp_A + vect_size, matrix_A.begin());

    auto *tmp_B = reinterpret_cast<int *>(taskData->inputs[1]);
    std::copy(tmp_B, tmp_B + vect_size, matrix_B.begin());
  }

  return true;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  // Getting the rank and size of the communicator
  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Sending the size of the matrixes to all processes
  int send_size = size;
  MPI_Bcast(&send_size, 4, MPI_INT, 0, MPI_COMM_WORLD);
  size = send_size;
  int vect_size = size * size;

  // Distribution of matrixes A and B to all processes
  if (world.rank() != 0) {
    matrix_A = std::vector<int>(vect_size);
    matrix_B = std::vector<int>(vect_size);
  }
  MPI_Bcast(matrix_A.data(), vect_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(matrix_B.data(), vect_size, MPI_INT, 0, MPI_COMM_WORLD);

  // Splitting matrix A into blocks and sending them
  int local_rows = size / world_size;       // the number of rows in each process
  int remainder_rows = size % world_size;   // residual rows distributed across the first processes
  int offset = 0;                           // shift counting
  std::vector<int> sendcounts(world_size);  // the number of elements for each process
  std::vector<int> displs(world_size);      // shift in the array of matrix A to transfer blocks

  for (int i = 0; i < world_size; ++i) {
    if (i < remainder_rows) {
      // Allocation of additional rows between the first processes
      sendcounts[i] = (local_rows + 1) * size;
    } else {
      sendcounts[i] = local_rows * size;
    }
    displs[i] = offset;
    offset += sendcounts[i];
  }
  // A local array for storing the submatrix A
  std::vector<int> local_A(sendcounts[world_rank]);
  // Sending matrix A rows to each process
  MPI_Scatterv(matrix_A.data(), sendcounts.data(), displs.data(), MPI_INT, local_A.data(), sendcounts[world_rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  // A local array for storing the res result
  int local_size = sendcounts[world_rank] / size;
  std::vector<int> local_res(local_size * size, 0);
  // Multiplying the local part of matrix A by matrix B
  for (int i = 0; i < local_size; ++i) {
    for (int j = 0; j < size; ++j) {
      local_res[i * size + j] = 0;
      for (int k = 0; k < size; ++k) {
        local_res[i * size + j] += local_A[i * size + k] * matrix_B[k * size + j];
      }
    }
  }

  // Combining the results of multiplication
  std::vector<int> recvcounts(world_size);  // The number of items from each process
  std::vector<int> recvdispls(world_size);  // Shifts for each process
  offset = 0;
  for (int i = 0; i < world_size; ++i) {
    if (i < remainder_rows) {
      // Allocation of additional rows between the first processes
      recvcounts[i] = (local_rows + 1) * size;
    } else {
      recvcounts[i] = local_rows * size;
    }
    recvdispls[i] = offset;
    offset += recvcounts[i];
  };
  // Collecting the multiplication results from each process in process 0
  MPI_Gatherv(local_res.data(), recvcounts[world_rank], MPI_INT, matrix_res.data(), recvcounts.data(),
              recvdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *ptr_matrix_res = reinterpret_cast<int *>(taskData->outputs[0]);
    std::copy(matrix_res.begin(), matrix_res.end(), ptr_matrix_res);
  }
  return true;
}