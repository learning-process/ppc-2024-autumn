#include "mpi/gusev_n_dijkstras_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/access.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace gusev_n_dijkstras_algorithm_mpi {
struct MinVertexComparator {
  std::vector<double> distances;

  MinVertexComparator() = default;

  MinVertexComparator(const std::vector<double>& dist) : distances(dist) {}

  int operator()(int a, int b) const {
    if (a == -1) return b;
    if (b == -1) return a;

    return (distances[a] < distances[b]) ? a : b;
  }
};

bool DijkstrasAlgorithmParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) return false;

  if (taskData->inputs.size() != taskData->inputs_count.size()) return false;

  auto* graph = reinterpret_cast<SparseGraphCRS*>(taskData->inputs[0]);

  if (taskData->inputs_count[0] != sizeof(SparseGraphCRS)) return false;

  if (graph->num_vertices == 0 || graph->num_vertices > 10000) return false;

  if (taskData->outputs.empty() || taskData->outputs[0] == nullptr) return false;

  if (taskData->outputs.size() != taskData->outputs_count.size()) return false;

  return true;
}

bool DijkstrasAlgorithmParallel::pre_processing() { return true; }

struct MinVertex {
  double distance;
  int vertex;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & distance;
    ar & vertex;
  }

  bool operator<(const MinVertex& other) const { return distance < other.distance; }
};

bool DijkstrasAlgorithmParallel::run() {
  auto* graph = reinterpret_cast<SparseGraphCRS*>(taskData->inputs[0]);
  int source_vertex = 0;
  int num_vertices = graph->num_vertices;
  int rank = world.rank();
  int num_procs = world.size();

  int vertices_per_proc = num_vertices / num_procs;
  int start_vertex = rank * vertices_per_proc;
  int end_vertex = (rank == num_procs - 1) ? num_vertices : start_vertex + vertices_per_proc;

  std::vector<double> local_distances(num_vertices, std::numeric_limits<double>::infinity());
  std::vector<bool> local_visited(num_vertices, false);

  if (rank == 0) {
    local_distances[source_vertex] = 0.0;
  }

  boost::mpi::broadcast(world, local_distances.data(), num_vertices, 0);

  for (int iter = 0; iter < num_vertices - 1; ++iter) {
    MinVertex local_min{std::numeric_limits<double>::infinity(), -1};

    for (int v = start_vertex; v < end_vertex; ++v) {
      if (!local_visited[v] && local_distances[v] < local_min.distance) {
        local_min.distance = local_distances[v];
        local_min.vertex = v;
      }
    }

    MinVertex global_min{std::numeric_limits<double>::infinity(), -1};
    boost::mpi::all_reduce(world, local_min, global_min,
                           [](const MinVertex& a, const MinVertex& b) { return a.distance < b.distance ? a : b; });

    if (global_min.vertex == -1 || global_min.distance == std::numeric_limits<double>::infinity()) break;

    local_visited[global_min.vertex] = true;

    for (int j = graph->row_ptr[global_min.vertex]; j < graph->row_ptr[global_min.vertex + 1]; ++j) {
      int neighbor = graph->col_indices[j];
      double weight = graph->values[j];

      if (neighbor >= start_vertex && neighbor < end_vertex && !local_visited[neighbor]) {
        double new_distance = local_distances[global_min.vertex] + weight;
        if (new_distance < local_distances[neighbor]) {
          local_distances[neighbor] = new_distance;
        }
      }
    }

    std::vector<double> global_distances(num_vertices);
    boost::mpi::all_reduce(world, local_distances.data(), num_vertices, global_distances.data(),
                           [](double a, double b) { return std::min(a, b); });

    local_distances = global_distances;
  }

  if (rank == 0) {
    auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(local_distances.begin(), local_distances.end(), output);
  }

  boost::mpi::broadcast(world, reinterpret_cast<double*>(taskData->outputs[0]), num_vertices, 0);

  return true;
}

bool DijkstrasAlgorithmParallel::post_processing() { return true; }
}  // namespace gusev_n_dijkstras_algorithm_mpi
