#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"

std::vector<int> createInputVector(size_t size, int initialValue = 0, int step = 1) {
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), initialValue);
  for (size_t i = 0; i < size; ++i) {
    vec[i] *= step;
  }
  return vec;
}


std::vector<int> calculateExpectedOutput(const std::vector<int>& input, size_t worldSize) {
  std::vector<int> output = input;
  size_t chunk_size = output.size() / worldSize;
  size_t remainder = output.size() % worldSize;

  for (size_t i = 0; i < worldSize; ++i) {
    size_t start_pos = i * chunk_size + std::min(i, remainder);
    size_t count = chunk_size + (i < remainder);
    for (size_t j = 0; j < count; ++j) {
      output[start_pos + j] += i;
    }
  }

  return output;
}

std::vector<int> generate_random_vector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> result(size);
  std::generate(result.begin(), result.end(), [&]() { return dist(gen); });
  return result;
}

TEST(anufriev_d_star_topology, EmptyVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(0);
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }
  anufriev_d_star_topology::SimpleIntMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.size(), static_cast<size_t>(0));
  }
}

TEST(anufriev_d_star_topology, SingleElementVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(1);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  anufriev_d_star_topology::SimpleIntMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, static_cast<size_t>(world.size()));
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, LargeVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(1000);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  anufriev_d_star_topology::SimpleIntMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, SimpleIntTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = {1, 2, 3, 4, 5};
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  anufriev_d_star_topology::SimpleIntMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    if (world.size() == 1) {
      ASSERT_EQ(output_data, std::vector<int>({1, 2, 3, 4, 5}));
    }
    if (world.size() == 2) {
      ASSERT_EQ(output_data, std::vector<int>({2, 4, 6, 8, 10}));
    }
    if (world.size() == 3) {
      ASSERT_EQ(output_data, std::vector<int>({3, 6, 9, 12, 15}));
    }
  }
}

TEST(anufriev_d_star_topology, SimpleIntTest_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = {1, 2, 3, 4, 5};
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  anufriev_d_star_topology::SimpleIntMPI task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (world.rank() == 0) {
    if (world.size() == 1) {
      ASSERT_EQ(output_data, std::vector<int>({1, 2, 3, 4, 5}));
    }
    if (world.size() == 2) {
      ASSERT_EQ(output_data, std::vector<int>({2, 4, 6, 8, 10}));
    }
    if (world.size() == 3) {
      ASSERT_EQ(output_data, std::vector<int>({3, 6, 9, 12, 15}));
    }
  }
}
