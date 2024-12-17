// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

using namespace zinoviev_a_bellman_ford_mpi;

TEST(zinoviev_a_bellman_ford, Test_Small_Graph) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_dist(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int num_vertices = 4;  // Количество вершин
    const int num_edges = 5;     // Количество рёбер
    global_graph = {0, 1, 1, 0, 2, 4, 1, 2, 2, 1, 3, 5, 2, 3, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
    taskDataPar->outputs_count.emplace_back(global_dist.size());
  }

  BellmanFordMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_dist = {0, 1, 3, 4};
    ASSERT_EQ(reference_dist, global_dist);
  }
}

TEST(zinoviev_a_bellman_ford, Test_Medium_Graph) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_dist(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int num_vertices = 5;  // Количество вершин
    const int num_edges = 7;     // Количество рёбер
    global_graph = {0, 1, 2, 0, 2, 4, 1, 2, 1, 1, 3, 7, 2, 3, 3, 2, 4, 5, 3, 4, 2};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
    taskDataPar->outputs_count.emplace_back(global_dist.size());
  }

  BellmanFordMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_dist = {0, 2, 3, 6, 8};
    ASSERT_EQ(reference_dist, global_dist);
  }
}