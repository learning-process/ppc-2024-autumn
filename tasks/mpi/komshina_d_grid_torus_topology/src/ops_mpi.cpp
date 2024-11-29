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
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0 || taskData->inputs[i] == nullptr) {
      return false;
    }
  }

  int rank = world.rank();
  int num_processes = world.size();

  int grid_size = std::sqrt(num_processes);
  if (grid_size * grid_size != num_processes) {
    return false;
  }

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::vector<int> input(input_data, input_data + taskData->inputs_count[0]);

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData == nullptr || taskData->outputs_count.empty()) {
      return false;
    }
  }
  int size = world.size();
  int sqrt_size = std::sqrt(size);
  return sqrt_size * sqrt_size == size;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::run() {
  try {
    compute_neighbors();

    std::vector<uint8_t> send_data(taskData->inputs_count[0]);
    std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

    std::vector<uint8_t> recv_data(taskData->inputs_count[0]);

    std::vector<boost::mpi::request> requests;
    requests.reserve(neighbors.size());

    for (int neighbor : neighbors) {
      requests.push_back(world.isend(neighbor, 0, send_data));
    }


    for (int neighbor : neighbors) {
      world.recv(neighbor, 0, recv_data);

      if (taskData->outputs_count[0] >= recv_data.size()) {
        std::copy(recv_data.begin(), recv_data.end(), taskData->outputs[0]);
      }
    }

    for (auto& req : requests) {
      req.wait();
    }

    world.barrier();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error during run: " << e.what() << std::endl;
    return false;
  }
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() {
  internal_order_test();

  if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->outputs.size(); ++i) {
    if (taskData->outputs_count[i] <= 0 || taskData->outputs[i] == nullptr) {
      return false;
    }
  }

  return true;
}

void komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::compute_neighbors() {
  int rank;
  const int x = rank % grid_size_x;
  const int y = rank / grid_size_x;

  const int left = ((x - 1 + grid_size_x) % grid_size_x) + y * grid_size_x;
  const int right = ((x + 1) % grid_size_x) + y * grid_size_x;
  const int up = x + ((y - 1 + grid_size_y) % grid_size_y) * grid_size_x;
  const int down = x + ((y + 1) % grid_size_y) * grid_size_x;

  neighbors = {left, right, up, down};
}
