#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mezhuev_m_lattice_torus {

bool GridTorusTopologyParallel::pre_processing() {
  if (!taskData) {
    return false;
  }

  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (!taskData->inputs[i] || taskData->inputs_count[i] <= 0) {
      return false;
    }
  }

  world.barrier();
  return true;
}

bool GridTorusTopologyParallel::validation() {
  if (!taskData || taskData->inputs.empty() || taskData->inputs_count.empty() || taskData->outputs.empty() ||
      taskData->outputs_count.empty()) {
    return false;
  }

  size_t total_input_size = 0;
  for (size_t i = 0; i < taskData->inputs_count.size(); ++i) {
    if (!taskData->inputs[i] || taskData->inputs_count[i] <= 0) {
      return false;
    }
    total_input_size += taskData->inputs_count[i];
  }

  size_t total_output_size = 0;
  for (size_t i = 0; i < taskData->outputs_count.size(); ++i) {
    if (!taskData->outputs[i] || taskData->outputs_count[i] <= 0) {
      return false;
    }
    total_output_size += taskData->outputs_count[i];
  }

  if (total_input_size != total_output_size) {
    return false;
  }

  int size = world.size();
  if (size == 1) {
    return true;
  }

  int grid_dim = static_cast<int>(std::sqrt(size));
  if (grid_dim * grid_dim != size) {
    return false;
  }

  return true;
}

bool GridTorusTopologyParallel::run() {
  if (!validation()) {
    return false;
  }

  int rank = world.rank();
  int size = world.size();

  if (size == 1) {
    std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], taskData->outputs[0]);
    return true;
  }

  int grid_dim = static_cast<int>(std::sqrt(size));

  int row = rank / grid_dim;
  int col = rank % grid_dim;

  int left = row * grid_dim + (col - 1 + grid_dim) % grid_dim;
  int right = row * grid_dim + (col + 1) % grid_dim;
  int up = ((row - 1 + grid_dim) % grid_dim) * grid_dim + col;
  int down = ((row + 1) % grid_dim) * grid_dim + col;

  std::vector<uint8_t> send_buffer(taskData->inputs_count[0]);
  std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_buffer.begin());

  std::vector<uint8_t> recv_left(taskData->inputs_count[0], 0);
  std::vector<uint8_t> recv_right(taskData->inputs_count[0], 0);
  std::vector<uint8_t> recv_up(taskData->inputs_count[0], 0);
  std::vector<uint8_t> recv_down(taskData->inputs_count[0], 0);

  try {
    world.send(left, 0, send_buffer);
    world.recv(left, 0, recv_left);

    world.send(right, 1, send_buffer);
    world.recv(right, 1, recv_right);

    world.send(up, 2, send_buffer);
    world.recv(up, 2, recv_up);

    world.send(down, 3, send_buffer);
    world.recv(down, 3, recv_down);
  } catch (const boost::mpi::exception& ex) {
    std::cerr << "MPI exception: " << ex.what() << std::endl;
    return false;
  }

  if (taskData->outputs_count[0] >= recv_left.size() + recv_right.size() + recv_up.size() + recv_down.size()) {
    std::copy(recv_left.begin(), recv_left.end(), taskData->outputs[0]);
    std::copy(recv_right.begin(), recv_right.end(), taskData->outputs[0] + recv_left.size());
    std::copy(recv_up.begin(), recv_up.end(), taskData->outputs[0] + recv_left.size() + recv_right.size());
    std::copy(recv_down.begin(), recv_down.end(),
              taskData->outputs[0] + recv_left.size() + recv_right.size() + recv_up.size());
  } else {
    return false;
  }

  world.barrier();
  return true;
}

bool GridTorusTopologyParallel::post_processing() {
  if (!taskData) {
    return false;
  }

  for (size_t i = 0; i < taskData->outputs.size(); ++i) {
    if (!taskData->outputs[i]) {
      return false;
    }

    for (size_t j = 0; j < taskData->outputs_count[i]; ++j) {
      if (taskData->outputs[i][j] == 0) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace mezhuev_m_lattice_torus
