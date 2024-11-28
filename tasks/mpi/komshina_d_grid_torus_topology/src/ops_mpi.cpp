#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::pre_processing() { return true; }

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  int size = world.size();
  int sqrt_size = std::sqrt(size);
  if (sqrt_size * sqrt_size != size) {
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
  try {
    int rank = world.rank();
    int size = world.size();

    int width = std::sqrt(size);
    int height = width;

    auto neighbors = compute_neighbors(rank, width, height);
    int left = neighbors[0];
    int right = neighbors[1];
    int up = neighbors[2];
    int down = neighbors[3];

    world.barrier();

    for (int step = 0; step < 1; ++step) {
      std::vector<uint8_t> send_data(taskData->inputs_count[0]);
      std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

      world.send(left, 0, send_data);
      world.send(right, 1, send_data);
      world.send(up, 2, send_data);
      world.send(down, 3, send_data);

      std::vector<uint8_t> recv_left(taskData->inputs_count[0]);
      std::vector<uint8_t> recv_right(taskData->inputs_count[0]);
      std::vector<uint8_t> recv_up(taskData->inputs_count[0]);
      std::vector<uint8_t> recv_down(taskData->inputs_count[0]);

      world.recv(left, 1, recv_left);
      world.recv(right, 0, recv_right);
      world.recv(up, 3, recv_up);
      world.recv(down, 2, recv_down);

      if (taskData->outputs_count[0] >= recv_left.size()) {
        std::copy(recv_left.begin(), recv_left.end(), taskData->outputs[0]);
      }
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    return false;
  }
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() { return true; }

std::array<int, 4> komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::compute_neighbors(int rank, int width,
                                                                                                    int height) {
  int left = (rank % width == 0) ? (rank + width - 1) : (rank - 1);
  int right = (rank % width == width - 1) ? (rank - width + 1) : (rank + 1);
  int up = (rank < width) ? (rank + width * (height - 1)) : (rank - width);
  int down = (rank >= width * (height - 1)) ? (rank - width * (height - 1)) : (rank + width);
  return {left, right, up, down};
