// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

TEST(koshkin_m_dining_philosophers, test_num_philosopher_1) {
  boost::mpi::communicator world;

  int num_philosophers = 0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  koshkin_m_dining_philosophers::testMpiTaskParallel testMpiTaskParallel(taskData);

  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(koshkin_m_dining_philosophers, test_num_philosopher_2) {
  boost::mpi::communicator world;

  int num_philosophers = 1;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  koshkin_m_dining_philosophers::testMpiTaskParallel testMpiTaskParallel(taskData);

  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(koshkin_m_dining_philosophers, test_num_philisophers_world) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  koshkin_m_dining_philosophers::testMpiTaskParallel testMpiTaskParallel(taskData);

  if (num_philosophers > 1) {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());
    bool deadlock_detected = testMpiTaskParallel.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(testMpiTaskParallel.validation());
}

class DiningPhilosophersTest : public ::testing::TestWithParam<int> {
 protected:
  boost::mpi::communicator world;
};

TEST_P(DiningPhilosophersTest, TestWithVariousPhilosophers) {
  int num_philosophers = GetParam();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  koshkin_m_dining_philosophers::testMpiTaskParallel testMpiTaskParallel(taskData);

  if (num_philosophers >= 2) {
    if (world.size() >= 2) {
      ASSERT_TRUE(testMpiTaskParallel.validation());
      ASSERT_TRUE(testMpiTaskParallel.pre_processing());
      ASSERT_TRUE(testMpiTaskParallel.run());
      ASSERT_TRUE(testMpiTaskParallel.post_processing());
      bool deadlock_detected = testMpiTaskParallel.check_deadlock();
      if (world.rank() == 0) {
        ASSERT_FALSE(deadlock_detected);
      }
    } else {
      ASSERT_FALSE(testMpiTaskParallel.validation());
    }
  } else {
    GTEST_SKIP();
  }
}
INSTANTIATE_TEST_SUITE_P(testMpiTaskParallel, DiningPhilosophersTest, ::testing::Values(2, 3, 4, 5, 6, 7, 10, 15, 17, 20, 30, 40, 60, 99));