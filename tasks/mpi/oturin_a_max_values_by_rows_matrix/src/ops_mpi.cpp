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
  /*
  if (world.size() <= 1) {  // triggerred on MSVC
    EXPECT_NE(1, 1) << "WORLD TOO SMALL" << std::endl;
    return false;
  }*/

  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0;
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

#if defined(_MSC_VER) && !defined(__clang__)
  if (world.size() == 1) {
    for (size_t i = 0; i < m; i++) {
      res[i] = *std::max_element(input_.begin() + i * n, input_.begin() + (i + 1) * n);
    }
    return true;
  }
#endif

#define TAG_EXIT 1
#define TAG_TOBASE 2
#define TAG_TOSAT 3

  /*
        m           maxes:
        ^
        | -9 99    :  99
        | 12 06    :  12
        +------> n
  */

  if (world.rank() == 0) {  // base
    int *arr = new int[m * n];
    int *maxes = new int[m];
    int exit[1] = {0};
    int noexit[1] = {1};

    std::copy(input_.begin(), input_.end(), arr);

    size_t satellites = world.size() - 1;

    size_t row = 0;
    while (row < m) {
      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        world.send(i + 1, TAG_EXIT, noexit, 1);
        world.send(i + 1, TAG_TOSAT, &arr[(row + i) * n], n);
      }

      for (size_t i = 0; i < std::min(satellites, m - row); i++) {
        world.recv(i + 1, TAG_TOBASE, &maxes[row + i], 1);
      }
      row += satellites;
    }
    for (size_t i = 0; i < satellites; i++)  // close all satellite processes
      world.send(i + 1, TAG_EXIT, exit, 1);

    res.assign(maxes, maxes + m);

    delete[] arr;
    delete[] maxes;
  } else {  // satelleite
    int *arr = new int[n];
    int *exit = new int[1];
    while (true) {
      int out = INT_MIN;
      world.recv(0, TAG_EXIT, exit, 1);
      if (exit[0] == 0) break;

      world.recv(0, TAG_TOSAT, arr, n);

      for (size_t i = 0; i < n; i++) out = std::max(arr[i], out);

      world.send(0, TAG_TOBASE, &out, 1);
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
  if (world.rank() == 0) {
    for (size_t i = 0; i < m; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
