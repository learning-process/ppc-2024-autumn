// Copyright 2023 Nesterov Alexander
#include <algorithm>
#include <random>
#include <vector>

#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

namespace zinoviev_a_bellman_ford_seq {

bool BellmanFordSeqTaskSequential::pre_processing() {
  internal_order_test();
  graph_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    graph_[i] = tmp_ptr[i];
  }
  dist_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool BellmanFordSeqTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] > 0;
}

bool BellmanFordSeqTaskSequential::run() {
  internal_order_test();
  return true;
}

bool BellmanFordSeqTaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (unsigned i = 0; i < dist_.size(); i++) {
    tmp_ptr[i] = dist_[i];
  }
  return true;
}

std::vector<int> generateRandomGraph(int num_vertices, int num_edges) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> vertex_dist(0, num_vertices - 1);
  std::uniform_int_distribution<> weight_dist(1, 10);

  std::vector<int> graph;
  for (int i = 0; i < num_edges; ++i) {
    int from = vertex_dist(gen);
    int to = vertex_dist(gen);
    int weight = weight_dist(gen);
    graph.push_back(from);
    graph.push_back(to);
    graph.push_back(weight);
  }
  return graph;
}

}  // namespace zinoviev_a_bellman_ford_seq