
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_topology_mpi, TestInsufficientData) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.clear();
  task_data->inputs_count.clear();
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation()) << "Validation should fail with insufficient input data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestValidation) {
  boost::mpi::communicator world;
  if (world.size() != 4) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  task_data->inputs.clear();
  task_data->inputs_count.clear();

  ASSERT_FALSE(task.validation());
}

TEST(komshina_d_grid_torus_topology_mpi, TestNonSquareTopology) {
  boost::mpi::communicator world;

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));

  if (sqrt_size * sqrt_size != size) {
    std::vector<uint8_t> input_data(4);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

    ASSERT_FALSE(task.validation()) << "Validation should fail for a non-square topology";
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestLargeData) {
  boost::mpi::communicator world;

  if (world.size() < 4) {
    GTEST_SKIP() << "There are not enough processes for this test";
  }

  size_t large_size = 1000;
  std::vector<uint8_t> input_data(large_size);
  std::iota(input_data.begin(), input_data.end(), 0);
  std::vector<uint8_t> output_data(large_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]) << i;
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestDataTransmission) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyOutputData) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 1);
  std::vector<uint8_t> output_data;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation()) << "Validation should fail with empty output data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestNullptrInput) {
  boost::mpi::communicator world;

  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation()) << "Validation should fail with nullptr in the input data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestPerformanceLargeData) {
  boost::mpi::communicator world;

  if (world.size() < 4) {
    GTEST_SKIP() << "There are not enough processes for the test";
  }

  size_t large_size = 10'000'000;
  std::vector<uint8_t> input_data(large_size, 42);
  std::vector<uint8_t> output_data(large_size, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output_data, input_data) << "The data does not match after transmission";
}

TEST(komshina_d_grid_torus_topology_mpi, TestErrorHandling) {
  boost::mpi::communicator world;

  if (world.size() < 4) {
    GTEST_SKIP() << "There are not enough processes for the test";
  }

  std::vector<uint8_t> input_data(4, 1);
  std::vector<uint8_t> output_data(4, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  if (world.rank() == 1) {
    task_data->inputs[0] = nullptr;
  }

  if (world.rank() != 1) {
    ASSERT_TRUE(task.validation()) << "Validation must take place for correct data";
  } else {
    ASSERT_FALSE(task.validation()) << "Validation should fail for incorrect data";
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestMPIExceptionHandling) {
  boost::mpi::communicator world;

  if (world.size() < 4) {
    GTEST_SKIP() << "There are not enough processes for the test";
  }

  std::vector<uint8_t> input_data(4, 42);
  std::vector<uint8_t> output_data(4, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  if (world.rank() == 1) {
    try {
      throw boost::mpi::exception("Mock exception", MPI_ERR_OTHER);
    } catch (const boost::mpi::exception& e) {
      std::cerr << "Caught mock MPI exception: " << e.what() << std::endl;
    }
  }

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}
