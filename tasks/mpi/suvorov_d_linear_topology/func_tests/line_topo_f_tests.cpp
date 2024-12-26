// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/suvorov_d_linear_topology/include/linear_topology.hpp"

namespace {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace

TEST(suvorov_d_linear_topology_mpi, test_with_normal_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 10000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  ASSERT_EQ(line_topo.validation(), true);
  line_topo.pre_processing();
  line_topo.run();
  line_topo.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(result_data);
  }
}

TEST(suvorov_d_linear_topology_mpi, test_with_large_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 1000000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  ASSERT_EQ(line_topo.validation(), true);
  line_topo.pre_processing();
  line_topo.run();
  line_topo.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(result_data);
  }
}

TEST(suvorov_d_linear_topology_mpi, test_with_prime_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 100003;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  ASSERT_EQ(line_topo.validation(), true);
  line_topo.pre_processing();
  line_topo.run();
  line_topo.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(result_data);
  }
}

TEST(suvorov_d_linear_topology_mpi, test_with_degree_of_two_vector_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 131072;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  ASSERT_EQ(line_topo.validation(), true);
  line_topo.pre_processing();
  line_topo.run();
  line_topo.post_processing();

  if (world.rank() == 0) {
    int result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}

TEST(suvorov_d_linear_topology_mpi, test_with_zero_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(line_topo.validation());
  }
}

TEST(suvorov_d_linear_topology_mpi, test_with_single_size) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  bool result_data = false;
  int count_size_vector = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    initial_data = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_data));
    taskDataPar->outputs_count.emplace_back(1);
  }

  suvorov_d_linear_topology_mpi::MPILinearTopology line_topo(taskDataPar);
  ASSERT_EQ(line_topo.validation(), true);
  line_topo.pre_processing();
  line_topo.run();
  line_topo.post_processing();

  if (world.rank() == 0) {
    int result = reinterpret_cast<int*>(taskDataPar->outputs[0])[0];
    EXPECT_TRUE(result);
  }
}
