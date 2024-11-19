// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/korovin_n_line_topology/include/ops_mpi.hpp"

TEST(korovin_n_line_topology_mpi, transfer_data) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP() << "There are not enough processes to run this test";
    return;
  }

  int n = 10000;
  auto root = 0;
  auto dst = world.size() - 1;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(root);
  taskData->inputs_count.emplace_back(dst);
  taskData->inputs_count.emplace_back(n);

  std::vector<int> data;
  std::vector<int> received_data;
  std::vector<int> received_trajectory;

  if (world.rank() == root) {
    data = korovin_n_line_topology_mpi::TestMPITaskParallel::generate_rnd_vector(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    world.send(dst, 0, data);
  }
  if (world.rank() == dst) {
    int trajectory_size = dst - root + 1;

    world.recv(root, 0, data);

    received_data.resize(n);
    received_trajectory.resize(trajectory_size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_trajectory.data()));
    taskData->outputs_count.emplace_back(received_trajectory.size());
  }

  korovin_n_line_topology_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == dst) {
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
    for (int i = 0; i < (int)received_trajectory.size(); i++) {
      ASSERT_EQ(received_trajectory[i], root + i);
    }
  }
}

TEST(korovin_n_line_topology_mpi, transfer_data_random) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP() << "There are not enough processes to run this test";
    return;
  }

  int n = 10000;

  std::srand(static_cast<unsigned int>(std::time(nullptr)) / 1000);
  int root = std::rand() % (world.size() - 1);

  int dst = (root + 1) + (world.size() > root + 1 ? std::rand() % (world.size() - (root + 1)) : 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(root);
  taskData->inputs_count.emplace_back(dst);
  taskData->inputs_count.emplace_back(n);

  std::vector<int> data;
  std::vector<int> received_data;
  std::vector<int> received_trajectory;

  if (world.rank() == root) {
    data = korovin_n_line_topology_mpi::TestMPITaskParallel::generate_rnd_vector(n);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    world.send(dst, 0, data);
  }
  if (world.rank() == dst) {
    int trajectory_size = dst - root + 1;

    world.recv(root, 0, data);

    received_data.resize(n);
    received_trajectory.resize(trajectory_size);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_data.data()));
    taskData->outputs_count.emplace_back(received_data.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_trajectory.data()));
    taskData->outputs_count.emplace_back(received_trajectory.size());
  }

  korovin_n_line_topology_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == dst) {
    for (int i = 0; i < n; i++) {
      ASSERT_EQ(received_data[i], data[i]);
    }
    for (int i = 0; i < (int)received_trajectory.size(); i++) {
      ASSERT_EQ(received_trajectory[i], root + i);
    }
  }
}
