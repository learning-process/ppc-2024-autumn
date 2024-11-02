#include "mpi/oturin_a_max_values_by_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  n = (size_t)(taskData->inputs_count[0]);
  m = (size_t)(taskData->inputs_count[1]);
  input_ = std::vector<int>(n * m);
  int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  input_ = std::vector<int>(tmp_ptr, tmp_ptr + n * m);
  // Init values for output
  res = std::vector<int>(m, 0);
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check elements count in i/o
  // m & maxes:
  return taskData->inputs_count[1] == taskData->outputs_count[0];
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
  if (world.rank() == 0) {
    return taskData->inputs_count[1] == taskData->outputs_count[0];
  }
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init vectors
  n = (size_t)(taskData->inputs_count[0]);
  m = (size_t)(taskData->inputs_count[1]);

  if (world.rank() == 0) {
    input_ = std::vector<int>(n * m);
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    input_ = std::vector<int>(tmp_ptr, tmp_ptr + n * m);
    // Init values for output
    res = std::vector<int>(m, 0);
  }

  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  const int TAG_EXIT = 1;
  const int TAG_TOBASE = 2;
  const int TAG_TOSAT = 3;

#if defined(_MSC_VER) && !defined(__clang__)
  if (world.size() == 1) {
    for (size_t i = 0; i < m; i++) {
      res[i] = *std::max_element(input_.begin() + i * n, input_.begin() + (i + 1) * n);
    }
    return true;
  }
#endif

  if (world.rank() == 0) {  // base
    size_t satellites = world.size() - 1;

    int proc_exit = 0;
    int proc_wait = 1;

    if (m == 0 || n == 0) {
      for (size_t i = 0; i < satellites; i++) {
        world.send(i + 1, TAG_EXIT, &proc_exit, 1);
      }
      return true;
    }

    int *arr = new int[m * n];
    int *maxes = new int[m];

    std::copy(input_.begin(), input_.end(), arr);

    size_t row = 0;
    while (row < m) {
      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        world.send(i + 1, TAG_EXIT, &proc_wait, 1);
        world.send(i + 1, TAG_TOSAT, &arr[(row + i) * n], n);
      }

      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        world.recv(i + 1, TAG_TOBASE, &maxes[row + i], 1);
      }
      row += satellites;
    }
    for (size_t i = 0; i < satellites; i++)  // close all satellite processes
      world.send(i + 1, TAG_EXIT, &proc_exit, 1);

    res.assign(maxes, maxes + m);

    delete[] arr;
    delete[] maxes;
  } else {  // satelleite
    int *arr = new int[n];
    int proc_exit;
    while (true) {
      int out = INT_MIN;
      world.recv(0, TAG_EXIT, &proc_exit, 1);
      if (proc_exit == 0) break;

      world.recv(0, TAG_TOSAT, arr, n);

      for (size_t i = 0; i < n; i++) out = std::max(arr[i], out);

      world.send(0, TAG_TOBASE, &out, 1);
    }
    delete[] arr;
  }

#undef TAG_EXIT
#undef TAG_TOBASE
#undef TAG_TOSAT
  return true;
}

bool oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < m; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
