#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"

#include <string>
#include <vector>

#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter_custom/include/ops_mpi.hpp"

bool varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_values.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_values.begin());
    res = 0;
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0 && taskData->outputs_count[0] != 1) {
    return false;
  }
  return world.size() >= 0 && world.rank() < world.size() && (ops == "+" || ops == "-" || ops == "max");
}

bool varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel::run() {
  internal_order_test();
  int node_size = 0;
  int local_res = 0;

  // Spread the data all over the processes. (including root)
  if (world.rank() == 0) {
    node_size = input_values.size() / world.size();
    local_input_values.resize(node_size + input_values.size() % world.size());
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, node_size);  // Sending node_size (root)
    }
  } else {  // Catching node_size (non-root)
    world.recv(0, 0, node_size);
    local_input_values.resize(node_size);
  }
  myScatter(world, input_values, local_input_values.data(), node_size, 0);
  if (world.rank() == 0) {
    std::copy(input_values.begin() + node_size * world.size(), input_values.end(),
              local_input_values.begin() + node_size);
  }
  if (ops == "+") {
    local_res = std::accumulate(local_input_values.begin(), local_input_values.end(), 0);
  } else if (ops == "-") {
    local_res = -std::accumulate(local_input_values.begin(), local_input_values.end(), 0);
  } else if (ops == "max") {
    if (!local_input_values.empty()) {
      local_res = local_input_values[0];
      for (int value : local_input_values) {
        if (value > local_res) {
          local_res = value;
        }
      }
    } else {
      return false;
    }
  }
  std::vector<int> results(world.size());
  if (world.rank() == 0) {
    results[0] = local_res;
    for (int i = 1; i < world.size(); i++) {
      world.recv(i, 0, results[i]);  // Sending results (root)
    }
  } else {  // Sending results (non-root)
    world.send(0, 0, local_res);
  }
  if (world.rank() == 0) {
    if (ops == "max") {
      if (!results.empty()) {
        res = results[0];
        for (int value : results) {
          if (value > res) {
            res = value;
          }
        }
      } else {
        return false;
      }
    } else {
      res = std::accumulate(results.begin(), results.end(), 0);
    }
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}