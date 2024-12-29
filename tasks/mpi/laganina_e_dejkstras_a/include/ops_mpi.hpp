#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <queue>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_dejskras_a_mpi {

std::vector<int> getRandomgraph(int v);
int minDistanceVertex(const std::vector<int>& dist, const std::vector<int>& marker);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  // static void dijkstra(int start_vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
  // const std::vector<int>& data, int v, std::vector<int>& distances);
  static void get_children_with_weights(int vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                                        const std::vector<int>& data, std::vector<int>& neighbor,
                                        std::vector<int>& weight) {
    // Get the beginning and end of edges for a given vertex
    int start = row_ptr[vertex];
    int end = row_ptr[vertex + 1];

    for (int i = start; i < end; ++i) {
      neighbor.push_back(col_ind[i]);  // Neighboring vertex
      weight.push_back(data[i]);       // Edge weight
    }
  }
  static void dijkstra(int start_vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                       const std::vector<int>& data, int v, std::vector<int>& distances) {
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

  // static void get_children_with_weights(int vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
  //  const std::vector<int>& data, std::vector<int>& neighbor,
  // std::vector<int>& weight);

 private:
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> data;
  int v{};  // dimension
  std::vector<int> distances;
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> data;
  int v{};  // dimension
  std::vector<int> distances;

  boost::mpi::communicator world;
};

}  // namespace laganina_e_dejskras_a_mpi