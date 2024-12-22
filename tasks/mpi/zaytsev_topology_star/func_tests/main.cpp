// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zaytsev_topology_star/include/ops_mpi.hpp"

TEST(zaytsev_topology_star, StarTopology_TrajectoryTest) {
  boost::mpi::communicator world;

  std::vector<int> test_vector = {1};

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = test_vector;
    expected_output.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(zaytsev_topology_star, StarTopology_PozitiveData) {
  boost::mpi::communicator world;

  std::vector<int> test_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = test_vector;
    expected_output.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(zaytsev_topology_star, StarTopology_NegativeData) {
  boost::mpi::communicator world;

  std::vector<int> test_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = test_vector;
    expected_output.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(zaytsev_topology_star, StarTopology_ZeroData) {
  boost::mpi::communicator world;

  std::vector<int> test_vector = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = test_vector;
    expected_output.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(zaytsev_topology_star, StarTopology_EmptyData) {
  boost::mpi::communicator world;

  std::vector<int> test_vector = {};

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = {0};
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(zaytsev_topology_star, StarTopology_LargeData) {
  boost::mpi::communicator world;

  std::vector<int> test_vector(1000, 1);

  size_t output_size = test_vector.size() + (world.size() * 2 - 1);
  std::vector<int> output_data(output_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_size);
  }

  zaytsev_topology_star::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_output = test_vector;
    expected_output.push_back(0);
    for (int i = 1; i < world.size(); ++i) {
      expected_output.push_back(i);
      expected_output.push_back(0);
    }

    ASSERT_EQ(output_data, expected_output);
  }
}