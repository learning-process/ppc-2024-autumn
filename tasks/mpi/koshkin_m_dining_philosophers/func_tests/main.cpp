// Copyright 2024 Koshkin Matvey
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
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

TEST(koshkin_m_dining_philosophers, test_num_philosopher_neg) {
  boost::mpi::communicator world;

  int num_philosophers = -2;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  koshkin_m_dining_philosophers::testMpiTaskParallel testMpiTaskParallel(taskData);

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
INSTANTIATE_TEST_SUITE_P(testMpiTaskParallel, DiningPhilosophersTest,
                         ::testing::Values(2, 3, 4, 5, 6, 7, 10, 15, 17, 20, 30, 40, 60, 99));

TEST(koshkin_m_dining_philosophers, test_random_num_philosophers_world) {
  boost::mpi::communicator world;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(200, 1000);
  int num_philosophers = dis(gen);

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

TEST(koshkin_m_dining_philosophers, test_check_deadlock) {
  boost::mpi::communicator world;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(world.size());
  int num_philosophers = world.size();

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

  bool deadlock_detected = testMpiTaskParallel.check_deadlock();

  if (world.rank() == 0) {
    ASSERT_FALSE(deadlock_detected);
  }

  testMpiTaskParallel.setStatus(2);
  world.barrier();

  deadlock_detected = testMpiTaskParallel.check_deadlock();

  if (world.rank() == 0) {
    ASSERT_TRUE(deadlock_detected);
  }
}
