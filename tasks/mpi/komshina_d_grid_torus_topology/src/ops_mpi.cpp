#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::pre_processing() {
  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0) {
      return false;
    }

    if (taskData->inputs[i] == nullptr) {
      return false;
    }
  }

  world.barrier();
  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
    return false;
  }
  if (taskData->outputs_count[0] < taskData->inputs_count[0]) {
    return false;
  }

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));
  if (sqrt_size * sqrt_size != size) {
    return false;
  }

  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();
  int grid_size = std::sqrt(size);

  world.barrier();

  int row = rank / grid_size;
  int col = rank % grid_size;

  int left = row * grid_size + (col - 1 + grid_size) % grid_size;
  int right = row * grid_size + (col + 1) % grid_size;
  int up = ((row - 1 + grid_size) % grid_size) * grid_size + col;
  int down = ((row + 1) % grid_size) * grid_size + col;

  std::vector<uint8_t> send_data(taskData->inputs_count[0]);
  std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

  world.send(left, 0, send_data);
  std::vector<uint8_t> recv_left(taskData->inputs_count[0]);
  world.recv(left, 0, recv_left);

  world.send(right, 1, send_data);
  std::vector<uint8_t> recv_right(taskData->inputs_count[0]);
  world.recv(right, 1, recv_right);

  world.send(up, 2, send_data);
  std::vector<uint8_t> recv_up(taskData->inputs_count[0]);
  world.recv(up, 2, recv_up);

  world.send(down, 3, send_data);
  std::vector<uint8_t> recv_down(taskData->inputs_count[0]);
  world.recv(down, 3, recv_down);

  if (taskData->outputs_count[0] >= send_data.size()) {
    std::copy(recv_left.begin(), recv_left.end(), taskData->outputs[0]);
  } else {
    std::cerr << "–азмер выходных данных недостаточен дл€ узла " << rank << std::endl;
  }

  world.barrier();
  return true;
}

bool komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::post_processing() { return true; }
