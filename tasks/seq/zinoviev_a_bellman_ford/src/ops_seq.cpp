// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace zinoviev_a_bellman_ford_seq {

void BellmanFordSeq::toCRS(const int* input_matrix) {
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

bool BellmanFordSeq::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  toCRS(input_matrix);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool BellmanFordSeq::validation() {
  internal_order_test();

  size_t V_count = taskData->inputs_count[0];
  size_t E_expected = taskData->inputs_count[1];
  size_t E_actual = 0;

  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);

  for (size_t i = 0; i < V_count * V_count; ++i) {
    if (input_matrix[i] != 0) {
      E_actual++;
    }
  }

  if (E_expected != E_actual) {
    return false;
  }

  for (size_t i = 0; i < V_count - 1; ++i) {
    bool has_outgoing = false;
    for (size_t j = 0; j < V_count; ++j) {
      if (input_matrix[i * V_count + j] != 0) {
        has_outgoing = true;
        break;
      }
    }
    if (!has_outgoing) {
      return false;
    }
  }

  for (size_t i = 0; i < V_count; ++i) {
    if (input_matrix[i * V_count + i] != 0) {
      return false;
    }
  }

  return V_count == taskData->outputs_count[0];
}

bool BellmanFordSeq::Iteration(std::vector<int>& paths) {
  bool changed = false;
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
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

bool BellmanFordSeq::check_negative_cycle() {
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (shortest_paths[i] != INF && shortest_paths[i] + weight < shortest_paths[v]) {
        return true;
      }
    }
  }
  return false;
}

bool BellmanFordSeq::run() {
  internal_order_test();

  bool changed = false;
  for (size_t i = 0; i < V - 1; ++i) {
    changed = Iteration(shortest_paths);
    if (!changed) break;
  }

  if (check_negative_cycle()) {
    return false;
  }

  for (size_t i = 0; i < V; ++i) {
    if (shortest_paths[i] == INF) shortest_paths[i] = 0;
  }
  return true;
}

bool BellmanFordSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
  }
  return true;
}

}  // namespace zinoviev_a_bellman_ford_seq