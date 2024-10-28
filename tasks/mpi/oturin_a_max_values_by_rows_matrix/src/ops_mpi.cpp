#include "mpi/oturin_a_max_values_by_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  n = (size_t)(taskData->inputs_count[0]);
  m = (size_t)(taskData->inputs_count[1]);
  input_ = std::vector<int>(n * m);
  int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (unsigned i = 0; i < n * m; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init values for output
  res = std::vector<int>(m, 0);
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of i/o
  // n && m && maxes:
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < m; i++) {
    res[i] = *std::max_element(input_.begin() + i * n, input_.begin() + (i + 1) * n);
  }
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < m; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
////////////////////////////////////////////////////////////////////////////////////////

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size <= 1) {
    EXPECT_EQ(1, 1) << "WORLD TOO SMALL" << std::endl;
    return false;
  }

  if (world_rank == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] > 1;
  }
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init vectors
  n = (size_t)(taskData->inputs_count[0]);
  m = (size_t)(taskData->inputs_count[1]);

  if (world_rank == 0) {
    input_ = std::vector<int>(n * m);
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    for (unsigned i = 0; i < n * m; i++) {
      input_[i] = tmp_ptr[i];
    }
    // Init values for output
    res = std::vector<int>(m, 0);
  }

  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

#define TAG_EXIT 1
#define TAG_TOBASE 2
#define TAG_TOSAT 3

  MPI_Status status;

  /*
        m           maxes:
        ^
        | -9 99    :  99
        | 12 06    :  12
        +------> n
  */

  if (world_rank == 0) {  // base
    int *arr = new int[m * n];
    int *maxes = new int[m];
    int exit[1] = {0};
    int noexit[1] = {1};

    std::copy(input_.begin(), input_.end(), arr);

    size_t satellites = world_size - 1;

    size_t row = 0;
    while (row < m) {
      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        MPI_Send(noexit, 1, MPI_INT, i + 1, TAG_EXIT, MPI_COMM_WORLD);
        MPI_Send(&arr[(row + i) * n], n, MPI_INT, i + 1, TAG_TOSAT, MPI_COMM_WORLD);
      }

      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        MPI_Recv(&maxes[row + i], 1, MPI_INT, i + 1, TAG_TOBASE, MPI_COMM_WORLD, &status);
      }
      row += satellites;
    }
    for (size_t i = 0; i < satellites; i++)  // close all satellite processes
      MPI_Send(exit, 1, MPI_INT, i + 1, TAG_EXIT, MPI_COMM_WORLD);

    res.assign(maxes, maxes + m);

    delete[] arr;
    delete[] maxes;
  } else {  // satelleite
    int *arr = new int[n];
    int *exit = new int[1];
    while (true) {
      int out = INT_MIN;
      MPI_Recv(exit, 1, MPI_INT, 0, TAG_EXIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (exit[0] == 0) break;
      MPI_Recv(arr, n, MPI_INT, 0, TAG_TOSAT, MPI_COMM_WORLD, &status);

      for (size_t i = 0; i < n; i++) out = std::max(arr[i], out);

      MPI_Send(&out, 1, MPI_INT, 0, TAG_TOBASE, MPI_COMM_WORLD);
    }
    delete[] arr;
    delete[] exit;
  }

#undef TAG_EXIT
#undef TAG_TOBASE
#undef TAG_TOSAT
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world_rank == 0) {
    for (size_t i = 0; i < m; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
