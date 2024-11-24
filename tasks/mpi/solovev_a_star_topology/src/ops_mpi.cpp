#include "mpi/solovev_a_star_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace solovev_a_star_topology_mpi {

std::vector<int> generate_random_vector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-1000, 1000);
  std::vector<int> random_vector(size);
  for (size_t i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }
  return random_vector;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }
  res = std::vector<int>(input_.size(), 0);
  order = std::vector<int>(world.size() + 1, -1);
  l_rank = -1;
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] == taskData->outputs_count[0]) && (taskData->inputs_count[0] > 0) &&
           (taskData->outputs_count[0] > 0);
  }
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    order.clear();
    order.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, input_);
      world.send(i, 0, l_rank);
    }
  } else {
    world.recv(0, 0, input_);
    world.recv(0, 0, l_rank);
    l_rank = world.rank();
    world.send(0, 0, input_);
    world.send(0, 0, l_rank);
  }
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      world.recv(i, 0, res);
      world.recv(i, 0, l_rank);
      order.push_back(l_rank);
    }
    order.push_back(world.size());
  }
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
    std::copy(order.begin(), order.end(), reinterpret_cast<int*>(taskData->outputs[1]));
  }
  return true;
}
}  // namespace solovev_a_star_topology_mpi
