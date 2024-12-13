// Copyright 2023 Nesterov Alexander
#include "mpi/leontev_n_gather/include/ops_mpi.hpp"

#include <cstdlib>
#include <string>

bool leontev_n_mat_vec_mpi::MPIMatVecSequential::pre_processing() {
  internal_order_test();
  int* vec_ptr1;
  int* vec_ptr2;
  if (taskData->inputs.size() >= 2) {
    vec_ptr1 = reinterpret_cast<int*>(taskData->inputs[0]);
    vec_ptr2 = reinterpret_cast<int*>(taskData->inputs[1]);
  } else {
    return false;
  }
  mat_ = std::vector<int>(taskData->inputs_count[0]);
  vec_ = std::vector<int>(taskData->inputs_count[1]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    mat_[i] = vec_ptr1[i];
  }
  for (size_t i = 0; i < taskData->inputs_count[1]; i++) {
    vec_[i] = vec_ptr2[i];
  }
  res = std::vector<int>(vec_.size(), 0);
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecSequential::validation() {
  internal_order_test();
  // Matrix+Vector input && vector input
  if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1) {
    return false;
  }
  // square matrix
  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }
  if (taskData->inputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    for (size_t j = 0; j < res.size(); j++) {
      res[i] += mat_[i * res.size() + j] * vec_[j];
    }
  }
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* vec_ptr1;
    int* vec_ptr2;
    if (taskData->inputs.size() >= 2) {
      vec_ptr1 = reinterpret_cast<int*>(taskData->inputs[0]);
      vec_ptr2 = reinterpret_cast<int*>(taskData->inputs[1]);
    } else {
      return false;
    }
    mat_ = std::vector<int>(taskData->inputs_count[0]);
    vec_ = std::vector<int>(taskData->inputs_count[1]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      mat_[i] = vec_ptr1[i];
    }
    for (size_t i = 0; i < taskData->inputs_count[1]; i++) {
      vec_[i] = vec_ptr2[i];
    }
    res = std::vector<int>(vec_.size(), 0);
  }
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1) {
      return false;
    }
    // square matrix
    if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
      return false;
    }
    if (taskData->inputs_count[0] == 0) {
      return false;
    }
  }
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecParallel::run() {
  internal_order_test();
  div_t divres;
  std::vector<int> local_input(res.size());
  if (world.rank() == 0) {
    divres = std::div(taskData->inputs_count[0], world.size());
  }
  broadcast(world, divres.quot, 0);
  if (world.rank() == 0) {
    boost::mpi::scatterv(world, local_input.data(), divres.quot + divres.rem, 0);
  } else {
    boost::mpi::scatterv(world, local_input.data(), divres.quot, 0);
  }
  for (size_t i = 0; i < res.size(); i++) {
    std::cerr << local_input[i] << std::endl;
  }
  res = std::vector<int>(local_input.size(), 0);
  return true;
}

bool leontev_n_mat_vec_mpi::MPIMatVecParallel::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
