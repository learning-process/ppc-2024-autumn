// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>

bool makhov_m_ring_topology::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init vector in root
  if (world.rank() == 0) {
    sequence.clear();
    input_data = std::vector<int>(taskData->inputs_count[0]);
    auto* data_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(data_ptr, data_ptr + taskData->inputs_count[0], input_data.begin());
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0 && taskData->outputs_count.size() == 2 &&
           taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.size() < 2) {
    output_data = input_data;
    sequence.push_back(0);
  }

  else {
    if (world.rank() == 0) {
      sequence.push_back(world.rank());
      world.send(world.rank() + 1, 0, input_data);
      world.send(world.rank() + 1, 1, sequence);

      int sender = world.size() - 1;
      world.recv(sender, 0, output_data);
      world.recv(sender, 1, sequence);
      sequence.push_back(world.rank());
    } else {
      int sender = world.rank() - 1;
      world.recv(sender, 0, input_data);
      world.recv(sender, 1, sequence);
      sequence.push_back(world.rank());

      int receiver = (world.rank() + 1) % world.size();
      world.send(receiver, 0, input_data);
      world.send(receiver, 1, sequence);
    }
  }
  return true;
}

bool makhov_m_ring_topology::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    int* sequence_ptr = reinterpret_cast<int*>(taskData->outputs[1]);

    std::copy(input_data.begin(), input_data.end(), output_ptr);
    std::copy(sequence.begin(), sequence.end(), sequence_ptr);
  }
  return true;
}
