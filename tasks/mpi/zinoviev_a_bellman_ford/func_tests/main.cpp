// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

TEST(zinoviev_a_bellman_ford, Test_Small_Graph) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_distances;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Example graph in CRS format
    global_graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    global_distances = std::vector<int>(4, INT_MAX);
    global_distances[0] = 0;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_distances.data()));
    taskDataPar->outputs_count.emplace_back(global_distances.size());
  }

  zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMPITaskParallel(taskDataPar);
  ASSERT_EQ(bellmanFordMPITaskParallel.validation(), true);
  bellmanFordMPITaskParallel.pre_processing();
  bellmanFordMPITaskParallel.run();
  bellmanFordMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Reference distances
    std::vector<int> reference_distances = {0, 2, 3, 1};
    ASSERT_EQ(global_distances, reference_distances);
  }
}

TEST(zinoviev_a_bellman_ford, Test_Medium_Graph) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_distances;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Example graph in CRS format
    global_graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    global_distances = std::vector<int>(6, INT_MAX);
    global_distances[0] = 0;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_distances.data()));
    taskDataPar->outputs_count.emplace_back(global_distances.size());
  }

  zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel bellmanFordMPITaskParallel(taskDataPar);
  ASSERT_EQ(bellmanFordMPITaskParallel.validation(), true);
  bellmanFordMPITaskParallel.pre_processing();
  bellmanFordMPITaskParallel.run();
  bellmanFordMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Reference distances
    std::vector<int> reference_distances = {0, 2, 3, 1, 2, 3};
    ASSERT_EQ(global_distances, reference_distances);
  }
}