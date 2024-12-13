// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/matyunina_a_dining_philosophers/include/ops_mpi.hpp"

TEST(matyunina_a_dining_philosophers_mpi, they_eat_thrice) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 3);
  std::vector<int32_t> average_value(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  matyunina_a_dining_philosophers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_EQ(testMpiTaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}

TEST(matyunina_a_dining_philosophers_mpi, they_eat_twice) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 2);
  std::vector<int32_t> average_value(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  matyunina_a_dining_philosophers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_EQ(testMpiTaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}

TEST(matyunina_a_dining_philosophers_mpi, they_eat_once) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 1);
  std::vector<int32_t> average_value(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  matyunina_a_dining_philosophers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_EQ(testMpiTaskParallel.validation(), false);
    }
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}
