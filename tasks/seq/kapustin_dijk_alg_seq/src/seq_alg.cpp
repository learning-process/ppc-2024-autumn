#include "seq/kapustin_dijk_alg_seq/include/seq_alg.hpp"

void kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential::CRSconvert(const int* input_matrix) {
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

bool kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  CRSconvert(input_matrix);

  res_.resize(V, INF);
  res_[0] = 0;
  return true;
}

bool kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) return false;
  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr) return false;
  return true;
}

bool kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential::run() {
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
bool kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}
