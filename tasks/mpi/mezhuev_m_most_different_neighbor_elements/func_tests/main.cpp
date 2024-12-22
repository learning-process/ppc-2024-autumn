#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "mpi/mezhuev_m_most_different_neighbor_elements/include/mpi.hpp"

TEST(mezhuev_m_most_different_neighbor_elements, ValidationEmptyTaskData) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationEmptyInputsOrOutputs) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(nullptr);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(10);
  taskData->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationInvalidSize) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(10);
  taskData->inputs_count.push_back(10);
  taskData->outputs_count.push_back(1);
  taskData->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationNullptrInInputsOrOutputs) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs.push_back(nullptr);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(10);
  taskData->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationEmptyCount) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));

  taskData->inputs_count.clear();
  taskData->outputs_count.clear();

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, ValidationSuccess) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(10);
  taskData->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_TRUE(task.validation());
}

TEST(mezhuev_m_most_different_neighbor_elements, PreProcessingZeroInputsCount) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(0);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
  ASSERT_FALSE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, PreProcessingInvalidInputsSize) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(10);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
  ASSERT_TRUE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, PreProcessingZeroDataSize) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(new int[1]));
  taskData->inputs_count.push_back(0);

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
  ASSERT_FALSE(task.pre_processing());
}

TEST(mezhuev_m_most_different_neighbor_elements, RunValidTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t data_size = 10;
  std::vector<int> input_data(data_size);
  std::iota(input_data.begin(), input_data.end(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.resize(1);

  taskData->inputs[0] = reinterpret_cast<uint8_t*>(input_data.data());
  taskData->inputs_count.resize(1);
  taskData->inputs_count[0] = data_size;

  taskData->outputs.resize(2);
  taskData->outputs[0] = reinterpret_cast<uint8_t*>(new int[1]);
  taskData->outputs[1] = reinterpret_cast<uint8_t*>(new int[1]);
  taskData->outputs_count.resize(1);
  taskData->outputs_count[0] = 1;

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  if (world.rank() == 0) {
    ASSERT_EQ(reinterpret_cast<int*>(taskData->outputs[0])[0], 0);
    ASSERT_EQ(reinterpret_cast<int*>(taskData->outputs[1])[0], 1);
  }
}

TEST(mezhuev_m_most_different_neighbor_elements, RunInvalidInputTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t data_size = 10;
  std::vector<int> input_data(data_size, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.resize(1);
  taskData->inputs[0] = reinterpret_cast<uint8_t*>(input_data.data());
  taskData->inputs_count.resize(1);
  taskData->inputs_count[0] = data_size;

  taskData->outputs.resize(2);
  taskData->outputs[0] = reinterpret_cast<uint8_t*>(new int[1]);
  taskData->outputs[1] = reinterpret_cast<uint8_t*>(new int[1]);
  taskData->outputs_count.resize(1);
  taskData->outputs_count[0] = 1;

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  if (world.rank() == 0) {
    ASSERT_EQ(reinterpret_cast<int*>(taskData->outputs[0])[0], 0);
    ASSERT_EQ(reinterpret_cast<int*>(taskData->outputs[1])[0], 0);
  }
}

TEST(mezhuev_m_most_different_neighbor_elements, RunWithNullPointersTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t data_size = 10;
  std::vector<int> input_data(data_size);
  std::iota(input_data.begin(), input_data.end(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.resize(1);
  taskData->inputs[0] = nullptr;
  taskData->inputs_count.resize(1);
  taskData->inputs_count[0] = data_size;

  taskData->outputs.resize(2);
  taskData->outputs[0] = nullptr;
  taskData->outputs[1] = nullptr;
  taskData->outputs_count.resize(1);
  taskData->outputs_count[0] = 1;

  mezhuev_m_most_different_neighbor_elements::MostDifferentNeighborElements task(world, taskData);

  ASSERT_FALSE(task.validation());
  ASSERT_TRUE(task.pre_processing());
}