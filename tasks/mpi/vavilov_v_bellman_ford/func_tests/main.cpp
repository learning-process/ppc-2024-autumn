#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_1) {
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();
    std::vector<int> edges = {0, 1, 10, 0, 2, 5, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 3, 9, 2, 4, 2, 3, 4, 4};
    std::vector<int> output(5);
    int vertices = 5, edges_count = 8, source = 0;
    taskDataPar->inputs_count.emplace_back(vertices);
    taskDataPar->inputs_count.emplace_back(edges_count);
    taskDataPar->inputs_count.emplace_back(source);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_TRUE(testTaskParallel.validation());
    ASSERT_TRUE(testTaskParallel.pre_processing());
    ASSERT_TRUE(testTaskParallel.run());
    ASSERT_TRUE(testTaskParallel.post_processing());

    std::vector<int> expected_output = {0, 8, 5, 9, 7};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_mpi, ValidInputWithMultiplePaths_2) {
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();
    std::vector<int> edges = {0, 1, -1, 0, 2, 4, 1, 2, 3, 1, 3, 2, 2, 3, 5, 3, 4, -3};
    std::vector<int> output(5);
    int vertices = 5, edges_count = 6, source = 0;

    taskDataPar->inputs_count.emplace_back(vertices);
    taskDataPar->inputs_count.emplace_back(edges_count);
    taskDataPar->inputs_count.emplace_back(source);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_TRUE(testTaskParallel.validation());
    ASSERT_TRUE(testTaskParallel.pre_processing());
    ASSERT_TRUE(testTaskParallel.run());
    ASSERT_TRUE(testTaskParallel.post_processing());

    std::vector<int> expected_output = {0, -1, 2, 1, -2};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_bellman_ford_seq, DisconnectedGraph) {
    mpi::communicator world;

    if (world.rank() == 0) {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();
      std::vector<int> edges = {0, 1, 4, 0, 2, 1, 1, 3, 2};
      std::vector<int> output(5);
      int vertices = 5, edges_count = 3, source = 0;

      taskDataPar->inputs_count.emplace_back(vertices);
      taskDataPar->inputs_count.emplace_back(edges_count);
      taskDataPar->inputs_count.emplace_back(source);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

      vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
      ASSERT_TRUE(testTaskParallel.validation());
      ASSERT_TRUE(testTaskParallel.pre_processing());
      ASSERT_TRUE(testTaskParallel.run());
      ASSERT_TRUE(testTaskParallel.post_processing());

      std::vector<int> expected_output = {0, 4, 1, 6, INT_MAX};
      EXPECT_EQ(output, expected_output);
    }
}

TEST(vavilov_v_bellman_ford_seq, NegativeCycle) {
    mpi::communicator world;

    if (world.rank() == 0) {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();
      std::vector<int> edges = {0, 1, 1, 1, 2, -1, 2, 0, -1};
      std::vector<int> output(3);
      int vertices = 3, edges_count = 3, source = 0;

      taskDataPar->inputs_count.emplace_back(vertices);
      taskDataPar->inputs_count.emplace_back(edges_count);
      taskDataPar->inputs_count.emplace_back(source);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

      vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
      ASSERT_TRUE(testTaskParallel.validation());
      ASSERT_TRUE(testTaskParallel.pre_processing());
      ASSERT_FALSE(testTaskParallel.run());
    }
}

TEST(vavilov_v_bellman_ford_seq, SingleVertexGraph) {
    mpi::communicator world;

    if (world.rank() == 0) {
      auto taskDataPar = std::make_shared<ppc::core::TaskData>();
      std::vector<int> edges = {};
      std::vector<int> output(1, 0);
      int vertices = 1, edges_count = 0, source = 0;

      taskDataPar->inputs_count.emplace_back(vertices);
      taskDataPar->inputs_count.emplace_back(edges_count);
      taskDataPar->inputs_count.emplace_back(source);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
      taskDataPar->outputs_count.emplace_back(output.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

      vavilov_v_bellman_ford_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
      ASSERT_TRUE(testTaskSequential.validation());
      ASSERT_TRUE(testTaskSequential.pre_processing());
      ASSERT_TRUE(testTaskSequential.run());
      ASSERT_TRUE(testTaskSequential.post_processing());

      std::vector<int> expected_output = {0};
      EXPECT_EQ(output, expected_output);
    }
}