#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace std::chrono_literals;

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::pre_processing() { return true; }

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0 || taskData->inputs[i] == nullptr) {
      return false;
    }
  }

  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }

  int size = boost::mpi::communicator().size();
  int sqrt_size = static_cast<int>(std::sqrt(size));
  return sqrt_size * sqrt_size == size;

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();

  double sqrt_size = std::sqrt(size);
  if (sqrt_size != static_cast<int>(sqrt_size)) {
    return false;
  }

  int grid_size = static_cast<int>(sqrt_size);

  world.barrier();

  std::vector<uint8_t> send_data(taskData->inputs_count[0], 0);
  std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

  for (int step = 0; step < grid_size; ++step) {
    auto neighbors = compute_neighbors(rank, grid_size);

    for (int neighbor : neighbors) {
      if (neighbor < size) {
        world.send(neighbor, 0, send_data);

        std::vector<uint8_t> recv_data(taskData->inputs_count[0]);
        world.recv(neighbor, 0, recv_data);

        if (taskData->outputs_count[0] >= send_data.size()) {
          std::copy(send_data.begin(), send_data.end(), taskData->outputs[0]);
        }
      }
    }

    world.barrier();
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() { return true; }

std::vector<int> komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::compute_neighbors(int rank,
                                                                                                  int grid_size) {
  int x = rank % grid_size;
  int y = rank / grid_size;

  int left = (x - 1 + grid_size) % grid_size + y * grid_size;
  int right = (x + 1) % grid_size + y * grid_size;

  int up = x + ((y - 1 + grid_size) % grid_size) * grid_size;
  int down = x + ((y + 1) % grid_size) * grid_size;

  return {left, right, up, down};
}