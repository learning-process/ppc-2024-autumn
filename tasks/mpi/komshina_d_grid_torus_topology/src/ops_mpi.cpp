#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::pre_processing() { 
  internal_order_test();

  if (world.rank() == 0) {
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }

  result = std::vector<int>(input_.size(), 0);
  order = std::vector<int>(world.size() + 1, -1);
  rank = -1;
  compute_neighbors();
  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->inputs_count[0] == taskData->outputs_count[0]) && (taskData->inputs_count[0] > 0) &&
           (taskData->outputs_count[0] > 0) && (world.size() > 1);
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::run() {
  internal_order_test();

  std::vector<int> data_to_send = input_;
  std::vector<int> received_data(input_.size());

  for (int step = 0; step < 4; ++step) {
    int neighbor = neighbors[step];

    if (neighbor != -1) {
      world.send(neighbor, 0, data_to_send);
      world.recv(neighbor, 0, received_data);

      for (size_t i = 0; i < result.size(); ++i) {
        result[i] += received_data[i];
      }
    }
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() {
  internal_order_test();
  world.barrier();

  if (world.rank() == 0) {
    std::copy(result.begin(), result.end(), reinterpret_cast<int*>(taskData->outputs[0]));
    std::copy(order.begin(), order.end(), reinterpret_cast<int*>(taskData->outputs[1]));
  }

  return true;
}

void komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::compute_neighbors() {
  const int size = world.size();
  int rows = 1;
  int cols = size;
  
  for (int i = 1; i <= std::sqrt(size); ++i) {
    if (size % i == 0) {
      rows = i;
      cols = size / i;
    }
  }

  const int row = rank / cols;
  const int col = rank % cols;

  neighbors = std::vector<int>(4, -1);
  neighbors[0] = (row > 0) ? rank - cols : rank + cols * (rows - 1);
  neighbors[1] = (row < rows - 1) ? rank + cols : rank - cols * (rows - 1);
  neighbors[2] = (col > 0) ? rank - 1 : rank + cols - 1;
  neighbors[3] = (col < cols - 1) ? rank + 1 : rank - (cols - 1);
}

