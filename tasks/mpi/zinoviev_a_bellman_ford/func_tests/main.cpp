// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

std::vector<int> createTestGraph(int num_vertices, const std::vector<std::pair<int, int>>& edges,
                                 const std::vector<int>& weights) {
  std::vector<int> graph = {num_vertices, static_cast<int>(edges.size())};
  std::vector<int> row_ptr(num_vertices + 1, 0);
  std::vector<int> col_ind;
  std::vector<int> values;

  // Build row pointers
  for (const auto& edge : edges) {
    row_ptr[edge.first + 1]++;
  }
  for (int i = 1; i <= num_vertices; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }

  // Build column indices and values
  for (size_t i = 0; i < edges.size(); i++) {
    col_ind.push_back(edges[i].second);
    values.push_back(weights[i]);
  }

  graph.insert(graph.end(), row_ptr.begin(), row_ptr.end());
  graph.insert(graph.end(), col_ind.begin(), col_ind.end());
  graph.insert(graph.end(), values.begin(), values.end());
  return graph;
}

TEST(zinoviev_a_bellman_ford, Test_Shortest_Path) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_dist(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_graph = 120;
    global_graph = zinoviev_a_bellman_ford_mpi::getRandomGraph(count_size_graph);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
    taskDataPar->outputs_count.emplace_back(global_dist.size());
  }

  zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMpiTaskParallel(taskDataPar);
  ASSERT_EQ(bellmanFordMpiTaskParallel.validation(), true);
  bellmanFordMpiTaskParallel.pre_processing();
  bellmanFordMpiTaskParallel.run();
  bellmanFordMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_dist(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataSeq->inputs_count.emplace_back(global_graph.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_dist.data()));
    taskDataSeq->outputs_count.emplace_back(reference_dist.size());

    // Create Task
    zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential bellmanFordMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(bellmanFordMpiTaskSequential.validation(), true);
    bellmanFordMpiTaskSequential.pre_processing();
    bellmanFordMpiTaskSequential.run();
    bellmanFordMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_dist[0], global_dist[0]);
  }

  TEST(zinoviev_a_bellman_ford, Test_Shortest_Path_Positive_Weights) {
    boost::mpi::communicator world;
    std::vector<int> global_graph;
    std::vector<int> global_dist(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      // Create a simple graph with 3 vertices and positive weights
      std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {0, 2}};
      std::vector<int> weights = {2, 3, 5};
      global_graph = createTestGraph(3, edges, weights);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
      taskDataPar->inputs_count.emplace_back(global_graph.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
      taskDataPar->outputs_count.emplace_back(global_dist.size());
    }

    zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMpiTaskParallel(taskDataPar);
    ASSERT_EQ(bellmanFordMpiTaskParallel.validation(), true);
    bellmanFordMpiTaskParallel.pre_processing();
    bellmanFordMpiTaskParallel.run();
    bellmanFordMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      // Create data
      std::vector<int> reference_dist(1, 0);

      // Create TaskData
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
      taskDataSeq->inputs_count.emplace_back(global_graph.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_dist.data()));
      taskDataSeq->outputs_count.emplace_back(reference_dist.size());

      // Create Task
      zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential bellmanFordMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(bellmanFordMpiTaskSequential.validation(), true);
      bellmanFordMpiTaskSequential.pre_processing();
      bellmanFordMpiTaskSequential.run();
      bellmanFordMpiTaskSequential.post_processing();

      // The shortest path from 0 to 2 should be 5 (0 -> 2)
      ASSERT_EQ(reference_dist[0], global_dist[0]);
      ASSERT_EQ(global_dist[0], 5);
    }
  }

  // Test case for a graph with negative weights
  TEST(zinoviev_a_bellman_ford, Test_Shortest_Path_Negative_Weights) {
    boost::mpi::communicator world;
    std::vector<int> global_graph;
    std::vector<int> global_dist(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      // Create a simple graph with 3 vertices and negative weights
      std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {0, 2}};
      std::vector<int> weights = {2, -3, 1};
      global_graph = createTestGraph(3, edges, weights);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
      taskDataPar->inputs_count.emplace_back(global_graph.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
      taskDataPar->outputs_count.emplace_back(global_dist.size());
    }

    zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMpiTaskParallel(taskDataPar);
    ASSERT_EQ(bellmanFordMpiTaskParallel.validation(), true);
    bellmanFordMpiTaskParallel.pre_processing();
    bellmanFordMpiTaskParallel.run();
    bellmanFordMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      // Create data
      std::vector<int> reference_dist(1, 0);

      // Create TaskData
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
      taskDataSeq->inputs_count.emplace_back(global_graph.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_dist.data()));
      taskDataSeq->outputs_count.emplace_back(reference_dist.size());

      // Create Task
      zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential bellmanFordMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(bellmanFordMpiTaskSequential.validation(), true);
      bellmanFordMpiTaskSequential.pre_processing();
      bellmanFordMpiTaskSequential.run();
      bellmanFordMpiTaskSequential.post_processing();

      // The shortest path from 0 to 2 should be -1 (0 -> 1 -> 2)
      ASSERT_EQ(reference_dist[0], global_dist[0]);
      ASSERT_EQ(global_dist[0], -1);
    }
  }

  // Test case for a graph with a negative cycle
  TEST(zinoviev_a_bellman_ford, Test_Negative_Cycle) {
    boost::mpi::communicator world;
    std::vector<int> global_graph;
    std::vector<int> global_dist(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      // Create a graph with a negative cycle
      std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
      std::vector<int> weights = {2, 3, -6};
      global_graph = createTestGraph(3, edges, weights);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
      taskDataPar->inputs_count.emplace_back(global_graph.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
      taskDataPar->outputs_count.emplace_back(global_dist.size());
    }

    zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMpiTaskParallel(taskDataPar);
    ASSERT_EQ(bellmanFordMpiTaskParallel.validation(), true);
    bellmanFordMpiTaskParallel.pre_processing();
    bellmanFordMpiTaskParallel.run();
    bellmanFordMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      // The algorithm should detect a negative cycle and return -1
      ASSERT_EQ(global_dist[0], -1);
    }
  }
