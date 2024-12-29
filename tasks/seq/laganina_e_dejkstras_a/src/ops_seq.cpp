#include "seq/laganina_e_dejkstras_a/include/ops_seq.hpp"

#include <thread>

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
