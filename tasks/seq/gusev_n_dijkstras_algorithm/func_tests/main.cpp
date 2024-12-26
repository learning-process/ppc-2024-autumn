// Copyright 2023 Nesterov Alexander
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "seq/gusev_n_dijkstras_algorithm/include/ops_seq.hpp"

TEST(gusev_n_dijkstras_algorithm_seq, SimpleGraphTest) {
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = 4;

  graph.row_ptr = {0, 2, 4, 6, 6};

  graph.col_indices = {1, 2, 0, 3, 1, 2};

  graph.values = {4.0, 2.0, 4.0, 3.0, 2.0, 1.0};

  std::vector<double> distances(4);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  std::vector<double> expected = {0.0, 4.0, 2.0, 7.0};
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(distances[i], expected[i], 1e-6);
  }
}

TEST(gusev_n_dijkstras_algorithm_seq, ComplexGraphTest) {
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = 5;

  graph.row_ptr = {0, 2, 4, 6, 8, 8};
  graph.col_indices = {1, 2, 0, 3, 2, 4, 1, 3};
  graph.values = {2.0, 4.0, 2.0, 1.0, 3.0, 5.0, 3.0, 2.0};

  std::vector<double> distances(5);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  std::vector<double> expected = {0.0, 2.0, 4.0, 3.0, 9.0};
  for (int i = 0; i < 5; ++i) {
    ASSERT_NEAR(distances[i], expected[i], 1e-6);
  }
}

TEST(gusev_n_dijkstras_algorithm_seq, DisconnectedGraphTest) {
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = 4;

  graph.row_ptr = {0, 1, 2, 2, 2};
  graph.col_indices = {1, 0};
  graph.values = {3.0, 3.0};

  std::vector<double> distances(4);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  ASSERT_EQ(distances[0], 0.0);
  ASSERT_EQ(distances[1], 3.0);
  ASSERT_TRUE(std::isinf(distances[2]));
  ASSERT_TRUE(std::isinf(distances[3]));
}
TEST(gusev_n_dijkstras_algorithm_seq, SingleVertexGraphTest) {
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = 1;
  graph.row_ptr = {0, 0};

  std::vector<double> distances(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  std::vector<double> expected = {0.0};
  ASSERT_NEAR(distances[0], expected[0], 1e-6);
}
TEST(gusev_n_dijkstras_algorithm_seq, EmptyGraphTest) {
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = 0;

  std::vector<double> distances(0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_FALSE(testTask->validation());
}
