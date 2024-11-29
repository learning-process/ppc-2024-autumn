// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <iostream>

#include "mpi\suvorov_d_linear_topology\include\linear_topology.hpp"

TEST(suvorov_d_linear_topology, test_with_normal_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> result_data(1, 0);
  int count_size_vector = 10;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = suvorov_d_linear_topology::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  suvorov_d_linear_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    bool result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}

TEST(suvorov_d_linear_topology, test_with_large_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> result_data(1, 0);
  int count_size_vector = 1000000;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = suvorov_d_linear_topology::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  suvorov_d_linear_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    bool result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}

TEST(suvorov_d_linear_topology, test_with_prime_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> result_data(1, 0);
  int count_size_vector = 100003;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = suvorov_d_linear_topology::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  suvorov_d_linear_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    bool result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}

TEST(suvorov_d_linear_topology, test_with_degree_of_two_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> result_data(1, 0);
  int count_size_vector = 131072;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = suvorov_d_linear_topology::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  suvorov_d_linear_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    bool result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}
