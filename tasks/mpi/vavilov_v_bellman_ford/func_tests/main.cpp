#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>

#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

static std::vector<int> generate_linear_graph(int num_vertices) {
  std::vector<int> matrix(num_vertices * num_vertices, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-20, 20);
  for (int i = 0; i < num_vertices - 1; ++i) {
    matrix[i * num_vertices + i + 1] = dist(gen);
  }

  return matrix;
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

TEST(vavilov_v_bellman_ford_mpi, Random_linear_graph) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int vertices = 300;
  int edges_count = 299;
  int source = 0;
  std::vector<int> output(vertices);
  std::vector<int> expected_output(vertices);
  std::vector<int> matrix = generate_linear_graph(vertices);
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
  int edges_count = 100;
  int source = 0;
  std::vector<int> output(vertices);
  std::vector<int> expected_output(vertices);
  std::vector<int> matrix = generate_random_sparse_graph(vertices, edges_count);
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
