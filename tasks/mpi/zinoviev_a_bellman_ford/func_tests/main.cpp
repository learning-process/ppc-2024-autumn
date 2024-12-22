// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

TEST(zinoviev_a_bellman_ford, Test_Small_Graph_mpi) {
  boost::mpi::communicator world;
  std::vector<int> graph = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<int> shortest_paths(4, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(3);
  taskData->inputs_count.emplace_back(4);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(3);

  zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = {0, 1, 0, 0};
    ASSERT_EQ(shortest_paths, expected);
  }
}

TEST(zinoviev_a_bellman_ford, Test_Medium_Graph_mpi) {
  boost::mpi::communicator world;
  std::vector<int> graph = {0, 4, 0, 0, 0, 0,  0, 8, 0, 4, 0,  8, 0, 0, 0,  0, 11, 0, 0, 8, 0, 7,  0,  4, 0, 0, 2,
                            0, 0, 7, 0, 9, 14, 0, 0, 0, 0, 0,  0, 9, 0, 10, 0, 0,  0, 0, 0, 4, 14, 10, 0, 2, 0, 0,
                            0, 0, 0, 0, 0, 2,  0, 1, 6, 8, 11, 0, 0, 0, 0,  1, 0,  7, 0, 0, 2, 0,  0,  0, 6, 7, 0};
  std::vector<int> shortest_paths(9, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(9);
  taskData->inputs_count.emplace_back(28);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(9);

  zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI task(taskData);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = {0, 4, 12, 19, 21, 11, 9, 8, 14};
    ASSERT_EQ(shortest_paths, expected);
  }
}