#include "mpi/milovankin_m_hypercube_topology/include/ops_mpi.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <vector>

bool milovankin_m_hypercube_topology::Hypercube::validation() {
  internal_order_test();

  // Number of processes must be a power of 2
  if ((world.size() & (world.size() - 1)) != 0) return false;
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs.size() != 1) return false;
    if (taskData->outputs.empty() || taskData->outputs_count.size() != 1) return false;
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* dataInPtr = reinterpret_cast<DataIn*>(taskData->inputs[0]);
    data_ = *dataInPtr;
    if (data_.destination >= world.size() || data_.destination <= 0) {
      return false;
    }
    data_.route.clear();
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::run() {
  internal_order_test();
  int world_size = world.size();
  int my_rank = world.rank();

  auto getNextId = [&world_size, &my_rank, &dest = this->data_.destination]() {
    for (uint16_t i = 0; i < std::log2(world_size); ++i) {
      uint16_t next = my_rank ^ (1 << i);
      if ((next ^ dest) < (my_rank ^ dest)) {
        return (int)next;
      }
    }
    return -1;
  };

  if (world.rank() == 0) {  // source process
    data_.route.push_back(0);
    int next = getNextId();
    world.send(next, 0, data_);
    world.recv(boost::mpi::any_source, 0, data_);
    // Send termination signal to unused processes
    data_.route[0] = -1;
    for (int i = 1; i < world.size(); ++i) {
      if (std::find(data_.route.begin(), data_.route.end(), i) == data_.route.end()) {
        world.send(i, 0, data_);
      }
    }
    data_.route[0] = 0;
  } else {
    // Recieve data, finish if it contains termination signal
    world.recv(boost::mpi::any_source, 0, data_);
    if (data_.route[0] == -1) {
      return true;
    }

    data_.route.push_back(world.rank());
    if (world.rank() != data_.destination) {  // intermediate process, calculate next and send
      int next = getNextId();
      if (next == -1) return false;
      world.send(next, 0, data_);
    } else {
      world.send(0, 0, data_);  // destination reached, send back to source process
    }
  }

  return true;
}

bool milovankin_m_hypercube_topology::Hypercube::post_processing() {
  internal_order_test();

  world.barrier();
  if (world.rank() == 0) {
    auto* dataOutPtr = reinterpret_cast<DataIn*>(taskData->outputs[0]);
    *dataOutPtr = data_;
  }

  return true;
}

// Calculate expected route from 0 to destination
std::vector<int> milovankin_m_hypercube_topology::Hypercube::calculate_route(int dest) {
  std::vector<int> route = {0};

  int current = 0;
  for (uint16_t i = 0; i <= std::log2(dest); ++i) {
    uint16_t next = current ^ (1 << i);  // flip i-th bit
    if ((next ^ dest) < (current ^ dest)) {
      route.push_back(next);
      current = next;
    }
  }

  return route;
}
