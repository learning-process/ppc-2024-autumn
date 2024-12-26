#include "mpi/tyurin_m_linear_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::validation() {
  internal_order_test();
  int val_rank = world.rank();
  int val_size = world.size();
  if (reinterpret_cast<int*>(taskData->inputs[0]) == nullptr || reinterpret_cast<int*>(taskData->inputs[1]) == nullptr)
    return false;
  int val_sender = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_target = *reinterpret_cast<int*>(taskData->inputs[1]);
  if (val_sender < 0 || val_target < 0 || std::max(val_sender, val_target) >= val_size || val_sender == val_target) {
    return false;
  }
  if (val_rank == val_sender) {
    return !(reinterpret_cast<int*>(taskData->inputs[2]) == nullptr);
  }
  if (val_rank == val_target) {
    return !(reinterpret_cast<bool*>(taskData->outputs[0]) == nullptr);
  }
  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::pre_processing() {
  internal_order_test();

  rank = world.rank();
  sender = *reinterpret_cast<int*>(taskData->inputs[0]);
  target = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (rank == sender) {
    data = *reinterpret_cast<std::vector<int>*>(taskData->inputs[2]);
    route.push_back(rank);
  } else if (rank == target) {
    result_flag = false;
  }

  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::run() {
  internal_order_test();

  if ((sender < target && (rank < sender || rank > target)) || (sender > target && (rank > sender || rank < target))) {
    return true;
  }

  if (rank == sender) {
    if (rank < target) {
      world.send(rank + 1, 0, std::make_pair(data, route));
    } else if (rank > target) {
      world.send(rank - 1, 0, std::make_pair(data, route));
    }
  } else {
    int source = (rank > sender) ? rank - 1 : rank + 1;
    std::pair<std::vector<int>, std::vector<int>> received;
    world.recv(source, 0, received);

    data = received.first;
    route = received.second;
    route.push_back(rank);

    if (rank != target) {
      if (rank < target) {
        world.send(rank + 1, 0, std::make_pair(data, route));
      } else {
        world.send(rank - 1, 0, std::make_pair(data, route));
      }
    } else {
      result_flag = true;
    }
  }

  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::post_processing() {
  internal_order_test();

  if (rank == target) {
    std::vector<int> shortest_route;
    if (sender < target) {
      for (int i = sender; i <= target; ++i) {
        shortest_route.push_back(i);
      }
    } else {
      for (int i = sender; i >= target; --i) {
        shortest_route.push_back(i);
      }
    }

    bool is_shortest = (route == shortest_route);
    *reinterpret_cast<bool*>(taskData->outputs[0]) = result_flag && is_shortest;
  }

  return true;
}
