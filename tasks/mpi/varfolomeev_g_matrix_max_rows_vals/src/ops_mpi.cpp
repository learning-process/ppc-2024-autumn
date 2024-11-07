#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <thread>

std::vector<int> varfolomeev_g_matrix_max_rows_vals_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> range(-100, 100);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

// Sequential
bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_m = taskData->inputs_count[0];  // rows count
  size_n = taskData->inputs_count[1];  // columns count
  mtr = std::vector<int>(size_m * size_n, 0);
  int *inpt_prt = reinterpret_cast<int *>(taskData->inputs[0]);
  mtr = std::vector<int>(inpt_prt, inpt_prt + size_n * size_m);
  res_vec.resize(size_m, 0);

  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] ==
         taskData->inputs_count[0];  // Checking that the number of output data is equal to the number of rows
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsSequential::run() {
  internal_order_test();
  for (int i = 0; i < size_m; i++) {
    res_vec[i] = *std::max_element(mtr.begin() + i * size_n, mtr.begin() + (i + 1) * size_n);
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

// Parallel
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
    return taskData->outputs_count[0] ==
           taskData->inputs_count[0];  // Checking that the number of output data is equal to the number of rows
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::run() {
  internal_order_test();
  // consts for data transm. between processes
  const int TERMINATE = 1;
  const int SEND_RESULT = 2;
  const int SEND_DATA = 3;

  if (world.size() == 1) {
    std::vector<int> local_maxes(size_m);
    for (int i = 0; i < size_m; ++i) {
      local_maxes[i] = *std::max_element(mtr.begin() + i * size_n, mtr.begin() + (i + 1) * size_n);
    }
    res_vec = local_maxes;
    return true;
  }

  if (world.rank() == 0) {  // root
    int exit_flag = 0;
    int continue_flag = 1;
    if (size_m == 0 || size_n == 0) {
      for (int i = 0; i < world.size() - 1; i++) {
        world.send(i + 1, TERMINATE, &exit_flag, 1);
      }
      return true;
    }

    std::vector<int> distr_vec(mtr.begin(), mtr.end());
    std::vector<int> max_vec(size_m);

    int row = 0;
    while (row < size_m) {
      for (int i = 0; i < std::min(world.size() - 1, size_m - row); i++) {
        world.send(i + 1, TERMINATE, &continue_flag, 1);
        world.send(i + 1, SEND_DATA, distr_vec.data() + (row + i) * size_n, size_n);
      }
      for (int i = 0; i < std::min(world.size() - 1, size_m - row); i++) {
        world.recv(i + 1, SEND_RESULT, max_vec.data() + row + i, 1);
      }
      row += world.size() - 1;
    }
    for (int i = 0; i < world.size() - 1; i++) {  // terminating all other processes
      world.send(i + 1, TERMINATE, &exit_flag, 1);
    }
    res_vec = max_vec;
  } else {  // not root
    std::vector<int> distr_vec(size_n);
    int exit_flag;
    while (true) {
      int out = INT_MIN;
      world.recv(0, TERMINATE, &exit_flag, 1);

      if (exit_flag == 0) break;

      world.recv(0, SEND_DATA, distr_vec.data(), size_n);
      out = *std::max_element(distr_vec.begin(), distr_vec.end());
      world.send(0, SEND_RESULT, &out, 1);
    }
  }
  return true;
}

bool varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < size_m; i++) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = res_vec[i];
    }
  }
  return true;
}
