#include "mpi/korotin_e_scatter/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool korotin_e_scatter_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(start, start + taskData->inputs_count[0], input_.begin());
  res = 0.0;
  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_scatter_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = *std::max_element(input_.begin(), input_.end());
  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == root) {
    input_ = std::vector<double>(taskData->inputs_count[0]);
    auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(start, start + taskData->inputs_count[0], input_.begin());
  }
  res = 0.0;
  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == root) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == root) {
    delta = taskData->inputs_count[0] / world.size();
    local_input_ = std::vector<double>(delta + taskData->inputs_count[0] % world.size());
    std::copy(input_.begin() + delta * world.size(), input_.end(), local_input_.begin() + delta);
  }

  broadcast(world, delta, root);

  if (world.rank() != root) {
    local_input_ = std::vector<double>(delta);
  }

  MPI_Scatter(input_.data(), delta, MPI_DOUBLE, local_input_.data(), delta, MPI_DOUBLE, root, MPI_COMM_WORLD);

  double local_res;

  if (local_input_.empty())
    local_res = -INFINITY;
  else
    local_res = *std::max_element(local_input_.begin(), local_input_.end());

  reduce(world, local_res, res, boost::mpi::maximum<double>(), root);

  return true;
}

bool korotin_e_scatter_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == root) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
