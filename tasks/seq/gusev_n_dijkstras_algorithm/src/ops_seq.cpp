#include "seq/gusev_n_dijkstras_algorithm/include/ops_seq.hpp"

bool gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) return false;
  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr) return false;

  auto* graph = reinterpret_cast<SparseGraphCRS*>(taskData->inputs[0]);

  return graph->num_vertices > 0 && graph->num_vertices <= 10000;
}

bool gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::run() {
  internal_order_test();

  auto* graph = reinterpret_cast<SparseGraphCRS*>(taskData->inputs[0]);
  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);

  int num_vertices = graph->num_vertices;
  int source_vertex = 0;

  std::vector<double> distances(num_vertices, std::numeric_limits<double>::infinity());
  std::vector<bool> visited(num_vertices, false);

  distances[source_vertex] = 0.0;

  for (int count = 0; count < num_vertices - 1; ++count) {
    double min_distance = std::numeric_limits<double>::infinity();
    int current_vertex = -1;

    for (int v = 0; v < num_vertices; ++v) {
      if (!visited[v] && distances[v] < min_distance) {
        min_distance = distances[v];
        current_vertex = v;
      }
    }

    if (current_vertex == -1) break;

    visited[current_vertex] = true;

    for (int j = graph->row_ptr[current_vertex]; j < graph->row_ptr[current_vertex + 1]; ++j) {
      int neighbor = graph->col_indices[j];
      double weight = graph->values[j];

      if (!visited[neighbor]) {
        double new_distance = distances[current_vertex] + weight;

        if (new_distance < distances[neighbor]) {
          distances[neighbor] = new_distance;
        }
      }
    }
  }

  std::copy(distances.begin(), distances.end(), output);
  return true;
}

bool gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::post_processing() {
  internal_order_test();
  return true;
}