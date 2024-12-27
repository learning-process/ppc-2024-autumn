#include "seq/laganina_e_dejkstras_a/include/ops_seq.hpp"

#include <queue>
#include <thread>
#include <vector>

bool laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::pre_processing() {
  internal_order_test();
  v = static_cast<int>(taskData->inputs_count[0]);
  std::vector<int> matrix_row(v * v, 0);
  for (int i = 0; i < v * v; i++) {
    matrix_row[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }

  int num_edges = 0;
  for (int i = 0; i < v * v; i++) {
    if (matrix_row[i] != 0) {
      num_edges++;
    }
  }
  row_ptr.resize(v + 1, 0);
  col_ind.resize(num_edges);
  data.resize(num_edges);
  int edge_index = 0;
  for (int i = 0; i < v; i++) {
    row_ptr[i] = edge_index;
    for (int j = 0; j < v; j++) {
      if (matrix_row[i * v + j] != 0) {
        col_ind[edge_index] = j;
        data[edge_index] = matrix_row[i * v + j];
        edge_index++;
      }
    }
  }
  row_ptr[v] = edge_index;
  return true;
}

bool laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::run() {
  internal_order_test();
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::dijkstra(0, row_ptr, col_ind, data, v, distances);
  return true;
}

bool laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < v; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = distances[i];
  }
  return true;
}

void laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq::dijkstra(int start_vertex, const std::vector<int>& row_ptr,
                                                                      const std::vector<int>& col_ind,
                                                                      const std::vector<int>& data, int v,
                                                                      std::vector<int>& distances) {
  // Initialize distances
  distances.resize(v, std::numeric_limits<int>::max());
  distances[start_vertex] = 0;

  // Array to track visited vertices
  std::vector<bool> visited(v, false);

  // Priority queue for storing pairs (distance, vertex)
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> priority_queue;
  priority_queue.emplace(0, start_vertex);  // Use start_vertex instead of 0

  while (!priority_queue.empty()) {
    int current_distance = priority_queue.top().first;
    int current_vertex = priority_queue.top().second;
    priority_queue.pop();

    // If the vertex has already been visited, skip it
    if (visited[current_vertex]) {
      continue;
    }

    // Mark the vertex as visited
    visited[current_vertex] = true;

    // Process all neighboring vertices
    int start = row_ptr[current_vertex];
    int end = row_ptr[current_vertex + 1];
    for (int i = start; i < end; ++i) {
      int neighbor_vertex = col_ind[i];
      int weight = data[i];
      int new_distance = current_distance + weight;

      // If a shorter distance is found, update it
      if (new_distance < distances[neighbor_vertex]) {
        distances[neighbor_vertex] = new_distance;
        priority_queue.emplace(new_distance, neighbor_vertex);
      }
    }
  }
}