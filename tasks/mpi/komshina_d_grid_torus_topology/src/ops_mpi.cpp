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
  if (taskData->inputs.empty()) {
    return false;
  }

  int size = world.size();
  int grid_size = std::sqrt(size);

  if (grid_size * grid_size != size) {
    return false;
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0) {
      return false;
    }

    if (taskData->inputs[i] == nullptr) {
      return false;
    }
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();
  int grid_size = std::sqrt(size);

  world.barrier();

  for (int step = 0; step < grid_size; ++step) {
    std::vector<int> neighbors = compute_neighbors(rank, grid_size);

    for (int neighbor : neighbors) {
      if (neighbor < size) {
        std::vector<uint8_t> send_data(taskData->inputs_count[0]);
        std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

        try {
          world.send(neighbor, 0, send_data);
          std::vector<uint8_t> recv_data(taskData->inputs_count[0]);
          world.recv(neighbor, 0, recv_data);

          if (taskData->outputs_count[0] >= send_data.size()) {
            std::copy(send_data.begin(), send_data.end(), taskData->outputs[0]);
          }
        } catch (const boost::mpi::exception& e) {
          std::cerr << "Error when exchanging data with the process " << neighbor << ": " << e.what() << std::endl;
        }
      }
    }
    world.barrier();
  }
  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() {
  int rank = world.rank();

  if (rank == 0) {
  }

  return true;
}

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
