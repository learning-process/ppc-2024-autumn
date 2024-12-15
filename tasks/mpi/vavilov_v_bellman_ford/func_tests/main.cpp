#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_1) {
  mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<int> matrix = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 9, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> output(5);
  int vertices = 5, edges_count = 8, source = 0;
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
  std::vector<int> row_offsets = {0, 2, 4, 5, 7, 8};
  std::vector<int> col_indices = {1, 2, 2, 3, 3, 4, 1, 2};
  std::vector<int> weights = {-1, 4, 3, 2, 5, -3, 2, -1};
  std::vector<int> output(5);
  int vertices = 5, edges_count = 6, source = 0;

  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_offsets.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indices.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(weights.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, -1, 2, 1, -2};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, DisconnectedGraph) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> row_offsets = {0, 2, 3, 3, 4, 4};
  std::vector<int> col_indices = {1, 2, 3};
  std::vector<int> weights = {4, 1, 2};
  std::vector<int> output(5);
  int vertices = 5, edges_count = 3, source = 0;

  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_offsets.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indices.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(weights.data()));
  taskDataPar->outputs_count.emplace_back(output.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  if (world.rank() == 0) {
    std::vector<int> expected_output = {0, 4, 1, 6, INT_MAX};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, NegativeCycle) {
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<int> row_offsets = {0, 1, 2, 3};
  std::vector<int> col_indices = {1, 2, 0};
  std::vector<int> weights = {1, -1, -1};
  std::vector<int> output(3);
  int vertices = 3, edges_count = 3, source = 0;

  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_offsets.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indices.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(weights.data()));
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
  std::vector<int> row_offsets = {0};
  std::vector<int> col_indices = {};
  std::vector<int> weights = {};
  std::vector<int> output(1, 0);
  int vertices = 1, edges_count = 0, source = 0;

  taskDataPar->inputs_count.emplace_back(vertices);
  taskDataPar->inputs_count.emplace_back(edges_count);
  taskDataPar->inputs_count.emplace_back(source);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_offsets.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indices.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(weights.data()));
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
