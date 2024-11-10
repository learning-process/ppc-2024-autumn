// Copyright 2023 Nesterov Alexander
#include "mpi/leontev_n_vector_sum/include/ops_mpi.hpp"

#include <cstdlib>
#include <string>

bool leontev_n_vec_sum_mpi::MPIVecSumSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* vec_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = vec_ptr[i];
  }
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool leontev_n_vec_sum_mpi::MPIVecSumSequential::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumParallel::run() {
  internal_order_test();
  std::div_t divres;

  if (world.rank() == 0) {
    divres = std::div(taskData->inputs_count[0], world.size());
  }

  broadcast(world, divres.quot, 0);
  broadcast(world, divres.rem, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    int* vec_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = vec_ptr[i];
    }

    for (int proc = 1; proc < world.size(); proc++) {
      int send_size = (proc == world.size() - 1) ? divres.quot + divres.rem : divres.quot;
      world.send(proc, 0, input_.data() + proc * divres.quot, send_size);
    }
  }
  local_input_ = std::vector<int>((world.rank() == world.size() - 1) ? divres.quot + divres.rem : divres.quot);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + divres.quot);
  } else {
    int recv_size = (world.rank() == world.size() - 1) ? divres.quot + divres.rem : divres.quot;
    world.recv(0, 0, local_input_.data(), recv_size);
  }
  int local_res;
  local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool leontev_n_vec_sum_mpi::MPIVecSumParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
