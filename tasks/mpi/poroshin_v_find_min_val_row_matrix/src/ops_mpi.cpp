// Copyright 2023 Nesterov Alexander
#include "mpi/poroshin_v_find_min_val_row_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> poroshin_v_find_min_val_row_matrix_mpi::gen(int m, int n) {
  std::vector<int> tmp(m * n);
  int n1 = std::max(n, m);
  int m1 = std::min(n, m);

  for (auto& t : tmp) {
    t = n1 + (std::rand() % (m1 - n1 + 7));
  }

  for (int i = 0; i < m; i++) {
    tmp[(std::rand() % n) + i * n] = INT_MIN;  // in 1 of n columns the value must be INT_MIN (needed to check answer)
  }

  return tmp;
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int m = taskData->inputs_count[0];
  int n = taskData->inputs_count[1];
  int size = m * n;
  input_.resize(size);
  res.resize(m);

  for (int i = 0; i < size; i++) {
    input_[i] = (reinterpret_cast<int*>(taskData->inputs[0])[i]);
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  int m = taskData->inputs_count[0];
  int n = taskData->inputs_count[1];

  for (int i = 0; i < m; i++) {
    int mn = INT_MAX;
    for (int j = n * i; j < n * i + n; j++) {
      mn = std::min(mn, input_[j]);
    }
    res[i] = mn;
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int n = 0;
  int m = 0;
  int size = 0;
  int delta = 0;

  if (world.rank() == 0) {
    m = taskData->inputs_count[0];
    n = taskData->inputs_count[1];
    size = n * m;
    delta = (size + world.size() - 1) / world.size();
  }

  broadcast(world, m, 0);
  broadcast(world, n, 0);
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_.resize(size);
    for (int i = 0; i < size; i++) {
      input_[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
    }
    local_input_.resize(delta);
    boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);
    res.resize(m, INT_MAX);
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            (taskData->outputs_count[0] == taskData->inputs_count[0]));
  } else {
    return true;
  }
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int n = 0;
  int m = 0;

  if (world.rank() == 0) {
    m = taskData->inputs_count[0];  // Number of rows
    n = taskData->inputs_count[1];  // Number of columns
  }

   broadcast(world, m, 0);
   broadcast(world, n, 0);

  int last = 0;

  if (world.rank() == world.size() - 1) {
    last = local_input_.size() * world.size() -
           n * m;  // Determine the number of elements that are not included in the last part
  }

  std::vector<int> res_(m, INT_MAX);  // Vector to store results (minimums)

  // Find local minima and save them in res_
  int delta = std::min(local_input_.size(), n - world.rank() * local_input_.size() % n);
  int local_min = *std::min_element(local_input_.begin(), local_input_.begin() + delta);
  res_[world.rank()] =
      local_min;  // Save the local minimum into the resulting vector at the index corresponding to the process rank
  int id = world.rank() + 1;  // Index to store results

  int k = 0;
  while (local_input_.begin() + delta + k * n < local_input_.end() - last) {
    local_min = *std::min_element(local_input_.begin() + delta + k * n,
                                  std::min(local_input_.end(), local_input_.begin() + delta + (k + 1) * n));
    if (id < m) {
      res_[id] = local_min;  // Save the local minimum into the resulting vector
    }
    k++;
    id++;
  }

  // Send results to process 0
  if (world.rank() != 0) {
    world.send(0, 0, res_);  // Send local minima
  } else {
    // Process 0 collects minimums from other processes
    for (int i = 0; i < world.size(); ++i) {
      if (i == 0) {
        // Save our minimums
        continue;
      }
      std::vector<int> received_res(m);
      world.recv(i, 0, received_res);  // Get minimums from other processes
      for (int j = 0; j < m; ++j) {
        if (received_res[j] < res_[j]) {
          res_[j] = received_res[j];  // Update global minimums
        }
      }
    }

    // Save the results
    for (int i = 0; i < m; ++i) {
      res[i] = res_[i];
    }
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    for (size_t i = 0; i < res.size(); i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }

  return true;
}