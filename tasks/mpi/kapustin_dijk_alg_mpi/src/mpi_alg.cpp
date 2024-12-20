#include "mpi/kapustin_dijk_alg_mpi/include/mpi_alg.hpp"
void kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ::CRSconvert(const int* input_matrix) {
  row_ptr.resize(V + 1);
  size_t non_zero_count = 0;

  for (size_t i = 0; i < V; ++i) {
    for (const int *row_start = input_matrix + i * V, *row_end = row_start + V; row_start != row_end; ++row_start) {
      if (*row_start != 0) {
        values.push_back(*row_start);
        columns.push_back(row_start - (input_matrix + i * V));
        ++non_zero_count;
      }
    }
    row_ptr[i + 1] = non_zero_count;
  }
}
bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  CRSconvert(input_matrix);

  res_.resize(V, INF);
  res_[0] = 0;
  return true;
}
bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) return false;
  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr) return false;
  return true;
}
bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ::run() {
  internal_order_test();
  std::vector<int> distances(V, INF);
  std::vector<bool> visited(V, false);
  std::set<std::pair<int, int>> active_vertices;

  distances[0] = 0;
  active_vertices.insert({0, 0});

  while (!active_vertices.empty()) {
    int u = active_vertices.begin()->second;
    active_vertices.erase(active_vertices.begin());

    if (visited[u]) continue;
    visited[u] = true;

    for (int j = row_ptr[u]; j < row_ptr[u + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      int new_dist = distances[u] + weight;

      if (new_dist < distances[v]) {
        active_vertices.erase({distances[v], v});
        distances[v] = new_dist;
        active_vertices.insert({new_dist, v});
      }
    }
  }

  for (size_t i = 0; i < V; ++i) {
    res_[i] = distances[i];
  }

  return true;
}
bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmSEQ::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

void kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::CRSconvert(const int* input_matrix) {
  row_ptr.push_back(0);
  for (size_t i = 0; i < V; ++i) {
    for (size_t j = 0; j < V; ++j) {
      if (input_matrix[i * V + j] != 0) {
        values.push_back(input_matrix[i * V + j]);
        columns.push_back(j);
      }
    }
    row_ptr.push_back(values.size());
  }

  values_size = values.size();
  columns_size = columns.size();
  row_ptr_size = row_ptr.size();
}

bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    V = taskData->inputs_count[0];
    E = taskData->inputs_count[1];

    CRSconvert(input_matrix);
  }

  return true;
}
bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) return false;
  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr) return false;
  return true;
}

bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, V, 0);
  boost::mpi::broadcast(world, E, 0);
  boost::mpi::broadcast(world, values_size, 0);
  boost::mpi::broadcast(world, row_ptr_size, 0);
  boost::mpi::broadcast(world, columns_size, 0);

  values.resize(values_size);
  row_ptr.resize(row_ptr_size);
  columns.resize(columns_size);

  boost::mpi::broadcast(world, values.data(), values.size(), 0);
  boost::mpi::broadcast(world, row_ptr.data(), row_ptr.size(), 0);
  boost::mpi::broadcast(world, columns.data(), columns.size(), 0);

  int delta = V / world.size();
  int extra = V % world.size();
  if (extra != 0) {
    delta += 1;
  }
  int start_index = world.rank() * delta;
  int end_index = std::min<int>(V, delta * (world.rank() + 1));

  res_.resize(V, INF);
  std::vector<bool> visited(V, false);

  if (world.rank() == 0) {
    res_[0] = 0;
  }

  boost::mpi::broadcast(world, res_.data(), V, 0);

  for (int k = 0; k < V; k++) {
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
                             return (a.second < b.second) ? a : b;
                           });
    if (global_pair.first == INF || global_pair.second == -1) {
      break;
    }

    visited[global_pair.second] = true;
    for (int j = row_ptr[global_pair.second]; j < row_ptr[global_pair.second + 1]; j++) {
      int v = columns[j];
      int w = values[j];

      if (!visited[v] && res_[global_pair.second] != INF &&
          (res_[global_pair.second] + w < res_[v])) {
        res_[v] = res_[global_pair.second] + w;
      }
    }

    std::vector<int> global_res_(V, INF);
    boost::mpi::all_reduce(world, res_.data(), V, global_res_.data(), boost::mpi::minimum<int>());
    res_ = global_res_;
  }

  return true;
}

bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < V; ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
  }

  return true;
}