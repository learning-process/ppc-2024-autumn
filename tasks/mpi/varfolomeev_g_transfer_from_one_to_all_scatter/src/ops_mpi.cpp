#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0;
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (ops == "+") {
    res = std::accumulate(input_.begin(), input_.end(), 0);
  } else if (ops == "-") {
    res = -std::accumulate(input_.begin(), input_.end(), 0);
  } else if (ops == "max") {
    res = *std::max_element(input_.begin(), input_.end());
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_values.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_values.begin());
  }
  res = 0;
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0 && taskData->outputs_count[0] != 1) {
    return false;
  }
  return world.size() >= 0 && world.rank() < world.size() && (ops == "+" || ops == "-" || ops == "max");
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int world_size = world.size();
  int local_res = 0;
  if (world.size() == 1) {
    if (ops == "+") {
      res = std::accumulate(input_values.begin(), input_values.end(), 0);
    } else if (ops == "-") {
      res = -std::accumulate(input_values.begin(), input_values.end(), 0);
    } else if (ops == "max") {
      res = input_values[0];
      for (int value : input_values) {
        if (value > res) {
          res = value;
        }
      }
    }
    return true;
  }
  // Spread the data
  if (world.rank() == 0) {
    for (int proc_num = 1; proc_num < world_size; proc_num++) {
      std::vector<int> local_data;
      for (int i = proc_num - 1; i < (int)input_values.size(); i += (world.size() - 1)) {
        local_data.push_back(input_values[i]);
      }
      world.send(proc_num, 0, local_data);
    }
  } else {
    world.recv(0, 0, local_input_values);
  }

  if (world.rank() != 0) {
    // Operation execution
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
  }

  if (world.rank() == 0) {  // catching results (root)
    std::vector<int> results(world_size - 1, -1);
    for (int proc_num = 1; proc_num < world_size; proc_num++) {
      int tmp;
      world.recv(proc_num, 0, tmp);
      results[proc_num - 1] = tmp;
    }
    if (ops == "max") {
      res = results[0];
      for (int value : results) {
        if (value > res) {
          res = value;
        }
      }
    } else {
      res = std::accumulate(results.begin(), results.end(), 0);
    }
  } else {  // Sending results (non-root)
    world.send(0, 0, local_res);
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::MyScatterTestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_values.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_values.begin());
    res = 0;
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::MyScatterTestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0 && taskData->outputs_count[0] != 1) {
    return false;
  }
  return world.size() >= 0 && world.rank() < world.size() && (ops == "+" || ops == "-" || ops == "max");
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::MyScatterTestMPITaskParallel::run() {
  internal_order_test();
  int node_size = 0;
  int local_res = 0;
  if (world.size() == 1) {
    if (ops == "+") {
      res = std::accumulate(input_values.begin(), input_values.end(), 0);
    } else if (ops == "-") {
      res = -std::accumulate(input_values.begin(), input_values.end(), 0);
    } else if (ops == "max") {
      res = input_values[0];
      for (int value : input_values) {
        if (value > res) {
          res = value;
        }
      }
    }
    return true;
  }

  if (world.rank() == 0) {
    node_size = input_values.size() / world.size();
    local_input_values.resize(node_size + input_values.size() % world.size());
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, node_size);
    }
  } else {
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
      world.recv(i, 0, results[i]);
    }
  } else {
    world.send(0, 0, local_res);
  }
  if (world.rank() == 0) {
    if (ops == "+") {
      res = std::accumulate(results.begin(), results.end(), 0);
    } else if (ops == "-") {
      res = std::accumulate(results.begin(), results.end(), 0);
    } else if (ops == "max") {
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
    }
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::MyScatterTestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}