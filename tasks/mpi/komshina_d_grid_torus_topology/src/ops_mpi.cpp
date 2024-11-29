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
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }
  result = std::vector<int>(input_.size(), 0);
  order = std::vector<int>(world.size() + 1, -1);
  rank = -1;

  int sqrt_size = static_cast<int>(std::sqrt(world.size()));
  width_x = length_y = sqrt_size;

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
  try {
    internal_order_test();

    int rank = world.rank();
    int left, right, up, down;

    compute_neighbors(rank, left, right, up, down);

    if (world.rank() == 0) {
      order.clear();
      order.push_back(0);

      world.send(right, 0, input_);
      world.send(down, 0, input_);
      world.send(left, 0, rank);
      world.send(up, 0, rank);
    } else {
      world.recv(left, 0, input_);
      world.recv(up, 0, rank);
      int my_rank = world.rank();
      world.send(right, 0, input_);
      world.send(down, 0, my_rank);
    }

    if (world.rank() == 0) {
      for (int i = 1; i < world.size(); ++i) {
        world.recv(i, 0, result);
        world.recv(i, 0, rank);
        order.push_back(rank);
      }
      order.push_back(world.size());
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    return false;
  }
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

void komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel::compute_neighbors(int rank, int& left, int& right,
                                                                                      int& up, int& down) {
  int x = rank % width_x;
  int y = rank / width_x;

  left = (x == 0) ? rank + width_x - 1 : rank - 1;
  right = (x == width_x - 1) ? rank - width_x + 1 : rank + 1;
  up = (y == 0) ? rank + (length_y - 1) * width_x : rank - width_x;
  down = (y == width_x - 1) ? rank - (length_y - 1) * width_x : rank + width_x;
}