#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>

#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

static std::vector<int> generate_random_tree(int vertices) {
  std::vector<int> graph(vertices * vertices, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> weight_dist(-20, 20);

  for (int i = 1; i < vertices; ++i) {
    int u = i;
    int v = std::uniform_int_distribution<int>(0, i - 1)(gen);
    int weight = weight_dist(gen);

    graph[u * vertices + v] = weight;
    graph[v * vertices + u] = weight;
  }

  return graph;
}

static std::vector<int> generate_random_complete_graph(int vertices) {
  std::vector<int> graph(vertices * vertices, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> weight_dist(-20, 20);

  for (int u = 0; u < vertices; ++u) {
    for (int v = u + 1; v < vertices; ++v) {
      int weight = weight_dist(gen);
      graph[u * vertices + v] = weight;
      graph[v * vertices + u] = weight;
    }
  }

  return graph;
}

static std::vector<int> generate_random_sparse_graph(int vertices, int edges_count) {
  std::vector<int> graph(vertices * vertices, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> weight_dist(-20, 20);
  std::uniform_int_distribution<int> vertex_dist(0, vertices - 1);

  int added_edges = 0;

  while (added_edges < edges_count) {
    int u = vertex_dist(gen);
    int v = vertex_dist(gen);

    if (u != v && graph[u * vertices + v] == 0) {
      graph[u * vertices + v] = weight_dist(gen);
      ++added_edges;
    }
  }

  return graph;
}

TEST(vavilov_v_bellman_ford_mpi, Random_tree) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int vertices = 100;
  int edges_count = 99;
  int source = 0;
  std::vector<int> output(vertices);
  std::vector<int> expected_output(vertices);
  std::vector<int> matrix = generate_random_tree(vertices);
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->outputs_count.emplace_back(expected_output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    vavilov_v_bellman_ford_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, Random_complete_graph) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int vertices = 100;
  int edges_count = 4950;
  int source = 0;
  std::vector<int> output(vertices);
  std::vector<int> expected_output(vertices);
  std::vector<int> matrix = generate_random_complete_graph(vertices);
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->outputs_count.emplace_back(expected_output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    vavilov_v_bellman_ford_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, Random_sparse_graph) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int vertices = 500;
  int edges_count = 515;
  int source = 0;
  std::vector<int> output(vertices);
  std::vector<int> expected_output(vertices);
  std::vector<int> matrix = generate_random_crs_graph(vertices, edges_count);
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->outputs_count.emplace_back(expected_output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    vavilov_v_bellman_ford_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_seq) {
  mpi::communicator world;
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<int> matrix = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 9, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  int vertices = 5;
  int edges_count = 8;
  int source = 0;
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  if (world.rank() == 0) {
    vavilov_v_bellman_ford_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    std::vector<int> expected_output = {0, 8, 5, 9, 7};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<int> matrix = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 9, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  std::vector<int> expected_output(5);
  int vertices = 5;
  int edges_count = 8;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->outputs_count.emplace_back(expected_output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    vavilov_v_bellman_ford_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_1) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<int> matrix = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 9, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  int vertices = 5;
  int edges_count = 8;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, 8, 5, 9, 7};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_2) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix = {0, -1, 4, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 5, -3, 2, 0, -1, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  int vertices = 5;
  int edges_count = 6;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, -1, 0, 1, -3};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, DisconnectedGraph) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix = {0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  int vertices = 5;
  int edges_count = 3;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, 4, 1, 3, INT_MAX};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, NegativeCycle) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix = {0, 1, 0, 0, 0, -1, -1, 0, 0};
  std::vector<int> output(3);
  int vertices = 3;
  int edges_count = 3;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_FALSE(testMpiTaskParallel.run());
}

TEST(vavilov_v_bellman_ford_mpi, SingleVertexGraph) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> matrix = {0};
  std::vector<int> output(1, 0);
  int vertices = 1;
  int edges_count = 0;
  int source = 0;
  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    std::vector<int> expected_output = {0};
    EXPECT_EQ(output, expected_output);
  }
}
