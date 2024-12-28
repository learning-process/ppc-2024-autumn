// Copyright 2023 Nesterov Alexander
#include "mpi/chizhov_m_algorithm_dijkstra/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

const int INF = std::numeric_limits<int>::max();

void chizhov_m_dijkstra_mpi::convertToCRS(const std::vector<int>& w, std::vector<int>& values,
                                          std::vector<int>& colIndex, std::vector<int>& rowPtr, int n) {
  rowPtr.resize(n + 1);
  int nnz = 0;
  for (int i = 0; i < n; i++) {
    rowPtr[i] = nnz;
    for (int j = 0; j < n; j++) {
      int weight = w[i * n + j];
      if (weight != 0) {
        values.emplace_back(weight);
        colIndex.emplace_back(j);
        nnz++;
      }
    }
  }
  rowPtr[n] = nnz;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  size = taskData->inputs_count[1];
  st = taskData->inputs_count[2];

  input_ = std::vector<int>(size * size);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);

  res_ = std::vector<int>(size, 0);
  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty()) {
    return false;
  }

  if (taskData->inputs_count.size() < 2 || taskData->inputs_count[1] <= 1) {
    return false;
  }

  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  if (!std::all_of(tmp_ptr, tmp_ptr + taskData->inputs_count[0], [](int val) { return val >= 0; })) {
    return false;
  }

  if (taskData->inputs_count[2] < 0 || taskData->inputs_count[2] >= taskData->inputs_count[1]) {
    return false;
  }

  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs.size() != 1 ||
      taskData->outputs_count[0] != taskData->inputs_count[1]) {
    return false;
  }
  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> values;
  std::vector<int> colIndex;
  std::vector<int> rowPtr;
  convertToCRS(input_, values, colIndex, rowPtr, size);

  std::vector<bool> visited(size, false);
  std::vector<int> D(size, INF);
  D[st] = 0;

  for (int i = 0; i < size; i++) {
    int min = INF;
    int index = -1;
    for (int j = 0; j < size; j++) {
      if (!visited[j] && D[j] < min) {
        min = D[j];
        index = j;
      }
    }

    if (index == -1) break;

    int u = index;
    visited[u] = true;

    for (int j = rowPtr[u]; j < rowPtr[u + 1]; j++) {
      int v = colIndex[j];
      int weight = values[j];

      if (!visited[v] && D[u] != INF && (D[u] + weight < D[v])) {
        D[v] = D[u] + weight;
      }
    }
  }

  res_ = D;

  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size = taskData->inputs_count[1];
    st = taskData->inputs_count[2];
  }

  if (world.rank() == 0) {
    input_ = std::vector<int>(size * size);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
    convertToCRS(input_, values, colIndex, rowPtr, size);
  } else {
    input_ = std::vector<int>(size * size, 0);
  }
  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty()) {
      return false;
    }

    if (taskData->inputs_count.size() < 2 || taskData->inputs_count[1] <= 1) {
      return false;
    }

    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    if (!std::all_of(tmp_ptr, tmp_ptr + taskData->inputs_count[0], [](int val) { return val >= 0; })) {
      return false;
    }

    if (taskData->inputs_count[2] < 0 || taskData->inputs_count[2] >= taskData->inputs_count[1]) {
      return false;
    }

    if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs.size() != 1 ||
        taskData->outputs_count[0] != taskData->inputs_count[1]) {
      return false;
    }
  }
  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, size, 0);
  boost::mpi::broadcast(world, st, 0);

  // broadcast of CRS vectors
  int values_size = values.size();
  int rowPtr_size = rowPtr.size();
  int colIndex_size = colIndex.size();

  boost::mpi::broadcast(world, values_size, 0);
  boost::mpi::broadcast(world, rowPtr_size, 0);
  boost::mpi::broadcast(world, colIndex_size, 0);

  values.resize(values_size);
  rowPtr.resize(rowPtr_size);
  colIndex.resize(colIndex_size);

  boost::mpi::broadcast(world, values.data(), values.size(), 0);
  boost::mpi::broadcast(world, rowPtr.data(), rowPtr.size(), 0);
  boost::mpi::broadcast(world, colIndex.data(), colIndex.size(), 0);

  int delta = size / world.size();
  int extra = size % world.size();
  if (extra != 0) {
    delta += 1;
  }
  int start_index = world.rank() * delta;
  int end_index = std::min(size, delta * (world.rank() + 1));

  res_.resize(size, INT_MAX);
  std::vector<bool> visited(size, false);
  std::vector<int> D(size, INF);

  if (world.rank() == 0) {
    res_[st] = 0;
  }

  boost::mpi::broadcast(world, res_.data(), size, 0);

  for (int k = 0; k < size; k++) {
    int local_min = INF;
    int local_index = -1;

    for (int i = start_index; i < end_index; i++) {
      if (!visited[i] && res_[i] < local_min) {
        local_min = res_[i];
        local_index = i;
      }
    }

    std::pair<int, int> local_pair = {local_min, local_index};
    std::pair<int, int> global_pair = {INF, -1};

    boost::mpi::all_reduce(world, local_pair, global_pair,
                           [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                             if (a.first < b.first) return a;
                             if (a.first > b.first) return b;
                             return a;
                           });

    if (global_pair.first == INF || global_pair.second == -1) {
      break;
    }

    visited[global_pair.second] = true;

    for (int j = rowPtr[global_pair.second]; j < rowPtr[global_pair.second + 1]; j++) {
      int v = colIndex[j];
      int w = values[j];

      if (!visited[v] && res_[global_pair.second] != INF && (res_[global_pair.second] + w < res_[v])) {
        res_[v] = res_[global_pair.second] + w;
      }
    }

    boost::mpi::all_reduce(world, res_.data(), size, D.data(), boost::mpi::minimum<int>());
    res_ = D;
  }

  return true;
}

bool chizhov_m_dijkstra_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}