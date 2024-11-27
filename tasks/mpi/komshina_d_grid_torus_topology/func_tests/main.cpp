
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <cmath>

#include "mpi/komshina_d_grid_torus_topology/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_topology_mpi, TestInsufficientNodes) {
  boost::mpi::communicator world;

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));
  if (sqrt_size * sqrt_size != size) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.push_back(input_data.data());
    task_data->inputs_count.push_back(input_data.size());
    task_data->outputs.push_back(output_data.data());
    task_data->outputs_count.push_back(output_data.size());

    komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);
    ASSERT_FALSE(task.validation());
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestPreProcessing) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(taskDataPar);

  ASSERT_TRUE(task.pre_processing());

  taskDataPar->inputs_count[0] = 0;
  ASSERT_FALSE(task.pre_processing());
}

TEST(komshina_d_grid_torus_topology_mpi, TestValidation) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());

  task_data->inputs.clear();
  task_data->inputs_count.clear();
  ASSERT_FALSE(task.validation());
}

TEST(komshina_d_grid_torus_topology_mpi, TestPostProcessing) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_TRUE(task.post_processing());

  std::vector<uint8_t> expected_output = {9, 10, 11, 12};
  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], expected_output[i]);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestDataTransmission) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestLargeData) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  size_t large_size = 1000;
  std::vector<uint8_t> input_data(large_size);
  std::iota(input_data.begin(), input_data.end(), 0);
  std::vector<uint8_t> output_data(large_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_TRUE(task.post_processing());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestUniformValues) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  size_t uniform_size = 100;
  uint8_t uniform_value = 42;
  std::vector<uint8_t> input_data(uniform_size, uniform_value);
  std::vector<uint8_t> output_data(uniform_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_TRUE(task.post_processing());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], uniform_value);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestNonMatchingInputOutputSizes) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);

  std::vector<uint8_t> output_data(2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(komshina_d_grid_torus_topology_mpi, TestSingleElementInput) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(1);
  input_data[0] = 9;
  std::vector<uint8_t> output_data(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());
  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(output_data[0], input_data[0]);
}
