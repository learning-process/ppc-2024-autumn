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
  return !taskData->inputs.empty() && taskData->inputs[0] != nullptr && !taskData->outputs.empty() &&
         taskData->outputs[0] != nullptr;
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

  std::copy(distances.begin(), distances.end(), res_.begin());

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
  row_ptr.resize(V + 1);
  size_t non_zero_count = 0;

  for (size_t i = 0; i < V; ++i) {
    const int* row_start = input_matrix + i * V;
    const int* row_end = row_start + V;

    for (; row_start != row_end; ++row_start) {
      if (*row_start != 0) {
        values.push_back(*row_start);
        columns.push_back(row_start - (input_matrix + i * V));
        ++non_zero_count;
      }
    }
    row_ptr[i + 1] = non_zero_count;
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
  return !taskData->inputs.empty() && taskData->inputs[0] != nullptr && !taskData->outputs.empty() &&
         taskData->outputs[0] != nullptr;
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

  int vertices_per_process = V / world.size();
  int start_vertex_index = world.rank() * vertices_per_process;
  int end_vertex_index = (world.rank() == world.size() - 1) ? V : start_vertex_index + vertices_per_process;

  res_.resize(V, INF);
  std::vector<bool> visited(V, false);

  if (world.rank() == 0) {
    res_[0] = 0;
  }

  boost::mpi::broadcast(world, res_.data(), V, 0);

  for (size_t iteration = 0; iteration < V; iteration++) {
    int local_min_distance = INF;
    int local_min_vertex = -1;
    int current_vertex = start_vertex_index;

    while (current_vertex < end_vertex_index) {
      if (!visited[current_vertex] && res_[current_vertex] < local_min_distance) {
        local_min_distance = res_[current_vertex];
        local_min_vertex = current_vertex;
      }
      current_vertex++;
    }

    std::pair<int, int> local_min_pair = {local_min_distance, local_min_vertex};
    std::pair<int, int> global_min_pair = {INF, -1};

    boost::mpi::reduce(
        world, local_min_pair, global_min_pair,
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
          if (a.first < b.first) return a;
          if (a.first > b.first) return b;
          return (a.second < b.second) ? a : b;
        },
        0);

    boost::mpi::broadcast(world, global_min_pair, 0);

    if (global_min_pair.first == INF) {
      break;
    }

    visited[global_min_pair.second] = true;

    for (int edge_index = row_ptr[global_min_pair.second]; edge_index < row_ptr[global_min_pair.second + 1];
         edge_index++) {
      int neighbor_vertex = columns[edge_index];
      int edge_weight = values[edge_index];

      if (!visited[neighbor_vertex] && res_[global_min_pair.second] != INF &&
          (res_[global_min_pair.second] + edge_weight < res_[neighbor_vertex])) {
        res_[neighbor_vertex] = res_[global_min_pair.second] + edge_weight;
      }
    }
  }
  std::vector<int> global_result(V, INF);
  boost::mpi::all_reduce(world, res_.data(), V, global_result.data(), boost::mpi::minimum<int>());
  res_ = global_result;

  return true;
}

bool kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_data);
  }

  return true;
}