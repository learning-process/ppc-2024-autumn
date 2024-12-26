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

  distances_.resize(vertices_, INT_MAX);
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
    std::for_each(edges_.begin(), edges_.end(), [this](const Edge& edge) {
      if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
        distances_[edge.dest] = distances_[edge.src] + edge.weight;
      }
    });
  }

  const bool has_negative_cycle = std::ranges::any_of(edges_, [this](const Edge& edge) {
    return distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest];
  });

  return !has_negative_cycle;
}

bool vavilov_v_bellman_ford_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(distances_.begin(), distances_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
