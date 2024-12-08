#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int n;
  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    input_ = std::vector<char>(n);
    output_ = std::vector<char>(n);
    std::copy(taskData->inputs[0], taskData->inputs[0] + n, input_.data());
    vector_size = n;
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->outputs_count.capacity() != 2 || taskData->inputs_count.size() != 1 ||
        taskData->inputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, vector_size, 0);
  int world_rank = world.rank();
  int world_size = world.size();
  if (world_size != 1) {
    if (world_rank == 0) {
      process_.push_back(0);
      world.send(world_rank + 1, 0, input_);
      world.send(world_rank + 1, 0, process_);
    } else {
      std::vector<char> buff(vector_size);
      int tmp_recv = world_rank - 1;
      int tmp_send = (world_rank + 1) % world_size;

      world.recv(tmp_recv, 0, buff);
      world.recv(tmp_recv, 0, process_);

      process_.push_back(world_rank);

      world.send(tmp_send, 0, buff);
      world.send(tmp_send, 0, process_);
    }
    if (world_rank == 0) {
      world.recv(world.size() - 1, 0, output_);
      process_.resize(world_size);
      world.recv(world.size() - 1, 0, process_);
    }
  } else {
    process_.push_back(0);
    std::copy(input_.data(), input_.data() + vector_size, output_.data());
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    for (int i = 0; i < vector_size; ++i) {
      reinterpret_cast<char*>(taskData->outputs[0])[i] = output_[i];
    }
    auto world_size = world.size();
    for (int i = 0; i < world_size; ++i) {
      reinterpret_cast<int*>(taskData->outputs[1])[i] = process_[i];
    }
    return true;
  }
  return true;
}
