#include "seq/vavilov_v_bellman_ford/include/ops_seq.hpp"

bool vavilov_v_bellman_ford_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  vertices_ = taskData->inputs_count[0];
  edges_count_ = taskData->inputs_count[1];
  source_ = taskData->inputs_count[2];
  int* edges_data = reinterpret_cast<int*>(taskData->inputs[0]);

  for (int i = 0; i < edges_count_; ++i) {
    edges_.push_back({edges_data[i * 3], edges_data[i * 3 + 1], edges_data[i * 3 + 2]});
  }

  distances_.resize(vertices_, INF);
  distances_[source_] = 0;

  return true;
}

bool vavilov_v_bellman_ford_seq::TestTaskSequential::validation() {
  internal_order_test();

  return (!taskData->inputs.empty());
}

bool vavilov_v_bellman_ford_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int i = 1; i < vertices_; ++i) {
    for (const auto& edge : edges_) {
      if (distances_[edge.src] != INF && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
        distances_[edge.dest] = distances_[edge.src] + edge.weight;
      }
    }
  }

  for (const auto& edge : edges_) {
    if (distances_[edge.src] != INF && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
      return false;  // Negative weight cycle detected
    }
  }

  return true;
}

bool vavilov_v_bellman_ford_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(distances_.begin(), distances_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
