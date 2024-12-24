#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "core/perf/include/perf.hpp"
#include "mpi/mezhuev_m_most_different_neighbor_elements/include/mpi.hpp"

TEST(mezhuev_m_most_different_neighbor_elements, SmallDataTest) {
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

TEST(mezhuev_m_most_different_neighbor_elements, LargeDataTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const size_t data_size = 10000;
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