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
    if (taskData->inputs[i] == nullptr || taskData->inputs_count[i] <= 0) {
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
    if (taskData->inputs[i] == nullptr || taskData->inputs_count[i] <= 0) {
      return false;
    }
    total_input_size += taskData->inputs_count[i];
  }

  size_t total_output_size = 0;
  for (size_t i = 0; i < taskData->outputs_count.size(); ++i) {
    if (taskData->outputs[i] == nullptr || taskData->outputs_count[i] <= 0) {
      return false;
    }
    total_output_size += taskData->outputs_count[i];
  }

  if (total_input_size != total_output_size) {
    return false;
  }

  int size = world.size();
  int grid_dim = static_cast<int>(std::sqrt(size));
  return grid_dim * grid_dim == size;
}

bool GridTorusTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();
  int grid_dim = static_cast<int>(std::sqrt(size));

  world.barrier();

  auto compute_neighbors = [grid_dim](int rank) -> std::vector<int> {
    int x = rank % grid_dim;
    int y = rank / grid_dim;

    int left = (x - 1 + grid_dim) % grid_dim + y * grid_dim;
    int right = (x + 1) % grid_dim + y * grid_dim;
    int up = x + ((y - 1 + grid_dim) % grid_dim) * grid_dim;
    int down = x + ((y + 1) % grid_dim) * grid_dim;

    return {left, right, up, down};
  };

  auto neighbors = compute_neighbors(rank);

  std::vector<uint8_t> send_buffer(taskData->inputs_count[0]);
  std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_buffer.begin());

  std::vector<uint8_t> combined_buffer;
  combined_buffer.reserve(taskData->inputs_count[0] * neighbors.size());

  for (int neighbor : neighbors) {
    try {
      world.send(neighbor, 0, send_buffer);
      std::vector<uint8_t> recv_buffer(taskData->inputs_count[0]);
      world.recv(neighbor, 0, recv_buffer);
      combined_buffer.insert(combined_buffer.end(), recv_buffer.begin(), recv_buffer.end());
    } catch (const boost::mpi::exception& ex) {
      std::cerr << "Error communicating with neighbor " << neighbor << ": " << ex.what() << std::endl;
      return false;
    }
  }

  if (taskData->outputs_count[0] >= combined_buffer.size()) {
    std::copy(combined_buffer.begin(), combined_buffer.end(), taskData->outputs[0]);
  } else {
    std::cerr << "Not enough space to store the output data." << std::endl;
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
    if (taskData->outputs[i] == nullptr) {
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