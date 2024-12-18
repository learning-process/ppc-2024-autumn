#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

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
