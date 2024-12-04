#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

TEST(mezhuev_m_lattice_torus, InsufficientNodeCount) {
  boost::mpi::communicator world;

  int total_size = world.size();
  int side_length = static_cast<int>(std::sqrt(total_size));
  if (side_length * side_length != total_size) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);
    ASSERT_FALSE(task.validation());
  }
}

TEST(mezhuev_m_lattice_torus, DataTransferTest) {
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

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
}

TEST(mezhuev_m_lattice_torus, MismatchedInputOutputSizes) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);

  std::vector<uint8_t> output_data(2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_lattice_torus, InvalidTopologySize) {
  boost::mpi::communicator world;
  int total_size = world.size();
  int grid_dimension = static_cast<int>(std::sqrt(total_size));

  if (grid_dimension * grid_dimension != total_size) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

    ASSERT_FALSE(task.validation());
  }
}

TEST(mezhuev_m_lattice_torus, TestPreProcessing) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.pre_processing());
}

TEST(mezhuev_m_lattice_torus, TestLargeGridProcessing) {
  boost::mpi::communicator world;

  if (world.size() < 16) return;

  std::vector<uint8_t> input_data(16);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(16);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(mezhuev_m_lattice_torus, TestIterationOnMaxGridSize) {
  boost::mpi::communicator world;
  if (world.size() < 16) return;

  int max_size = 256;
  std::vector<uint8_t> input_data(max_size);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(max_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_TRUE(task.validation());

  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(mezhuev_m_lattice_torus, TestUnmatchedInputOutputSizesWithLargeData) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  size_t large_size = 1024 * 1024;
  std::vector<uint8_t> input_data(large_size);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(large_size / 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(mezhuev_m_lattice_torus, TestHandlingOfUnsupportedDataTypes) {
  boost::mpi::communicator world;
  if (world.size() < 2) return;

  std::vector<float> unsupported_input_data(4);
  std::iota(unsupported_input_data.begin(), unsupported_input_data.end(), 1.0f);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(unsupported_input_data.data()));
  task_data->inputs_count.emplace_back(unsupported_input_data.size() * sizeof(float));
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  mezhuev_m_lattice_torus::GridTorusTopologyParallel task(task_data);

  ASSERT_FALSE(task.validation());
}