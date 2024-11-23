#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/gusev_n_cybercube_topology/include/ops_mpi.hpp"

namespace gusev_n_cybercube_topology_mpi {

TEST(gusev_n_cybercube_topology_mpi, TestInsufficientNodes) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    std::vector<uint8_t> input_data = {1, 2, 3, 4};
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.push_back(input_data.data());
    task_data->inputs_count.push_back(input_data.size());
    task_data->outputs.push_back(output_data.data());
    task_data->outputs_count.push_back(output_data.size());

    CybercubeTopologyParallel task(task_data);

    ASSERT_FALSE(task.validation());
  }
}

TEST(gusev_n_cybercube_topology_mpi, TestDataTransmission) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data = {1, 2, 3, 4};
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  CybercubeTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());

  try {
    ASSERT_TRUE(task.run());
  } catch (const std::exception& e) {
    std::cerr << "Exception caught during run: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_TRUE(task.post_processing());

  std::vector<uint8_t> expected_output = {1, 2, 3, 4};
  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], expected_output[i]);
  }
}

TEST(gusev_n_cybercube_topology_mpi, TestPreProcessing) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data = {5, 6, 7, 8};
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  CybercubeTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
}

TEST(gusev_n_cybercube_topology_mpi, TestPostProcessing) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data = {9, 10, 11, 12};
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  CybercubeTopologyParallel task(task_data);
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

}  // namespace gusev_n_cybercube_topology_mpi
