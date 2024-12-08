// Copyright 2024 Nesterov Alexander
#include "seq/gnitienko_k_bellman-ford_algorithm/include/ops_seq.hpp"

void gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::toCRS(const int* input_matrix) {
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
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  toCRS(input_matrix);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] < taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::Iteration(std::vector<int>& paths) {
  bool changed = false;
  for (size_t i = 0; i < V; ++i) {
    for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (paths[i] != INF && paths[i] + weight < paths[v]) {
        paths[v] = paths[i] + weight;
        changed = true;
      }
    }
  }

  return changed;
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::check_negative_cycle() {
  for (size_t i = 0; i < V; ++i) {
    for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (shortest_paths[i] != INF && shortest_paths[i] + weight < shortest_paths[v]) {
        std::cerr << "Negative cycle detected!" << std::endl;
        return true;
      }
    }
  }
  return false;
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::run() {
  internal_order_test();

  bool changed = false;
  for (int i = 0; i < V - 1; i++) {
    changed = Iteration(shortest_paths);
    if (!changed) break;
  }

  if (check_negative_cycle()) {
    return false;
  }

  for (int i = 0; i < V; i++) {
    if (shortest_paths[i] == INF) shortest_paths[i] = 0;
  }
  return true;
}

bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::post_processing() {
  internal_order_test();
  for (int i = 0; i < V; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
  }
  return true;
}

//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::pre_processing() {
//  internal_order_test();
//  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
//  V = taskData->inputs_count[0];
//  E = taskData->inputs_count[1];
//  adjacency_matrix.resize(V, std::vector<int>(V));
//  for (int i = 0; i < V; i++) {
//    for (int j = 0; j < V; j++) {
//      adjacency_matrix[i][j] = input_matrix[i * V + j];
//    }
//  }
//  shortest_paths.resize(V, (INT_MAX - 1)/2);
//  shortest_paths[0] = 0;
//  return true;
//}
//
//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::validation() {
//  internal_order_test();
//  return taskData->inputs_count[0] < taskData->inputs_count[1] &&
//         taskData->inputs_count[0] == taskData->outputs_count[0];
//}
//
//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::Iteration(std::vector<int>& paths) {
//  bool changed = false;
//  for (int i = 0; i < V; i++) {
//    for (int j = 0; j < V; j++) {
//      if (i != j) {
//        if (adjacency_matrix[i][j] != 0 && adjacency_matrix[i][j] + paths[i] < paths[j]) {
//          paths[j] = adjacency_matrix[i][j] + paths[i];
//          changed = true;
//        }
//      }
//    }
//  }
//
//  return changed;
//}
//
//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::check_negative_cycle() {
//  for (int i = 0; i < V; i++) {
//    for (int j = 0; j < V; j++) {
//      if (i != j && adjacency_matrix[i][j] != 0) {
//        if (shortest_paths[i] + adjacency_matrix[i][j] < shortest_paths[j]) {
//          std::cerr << "Negative cycle detected!" << std::endl;
//          return true;
//        }
//      }
//    }
//  }
//  return false;
//}
//
//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::run() {
//  internal_order_test();
//  bool stop = false;
//
//  for (int i = 0; i < V - 1; i++) {
//    stop = Iteration(shortest_paths);
//    if (!stop) break;
//  }
//
//  if (check_negative_cycle()) {
//    return false;
//  }
//  for (int i = 0; i < V; i++)
//    if (shortest_paths[i] == (INT_MAX - 1) / 2) shortest_paths[i] = NULL;
//
//  return true;
//}
//
//bool gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq::post_processing() {
//  internal_order_test();
//  for (int i = 0; i < V; i++) {
//    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
//  }
//  return true;
//}
