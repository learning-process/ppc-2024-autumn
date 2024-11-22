#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>

#include "mpi/beskhmelnova_k_dinning_philosophers/include/dinning_philosophers.hpp"
#include "mpi/beskhmelnova_k_dinning_philosophers/src/dinning_philosophers.cpp"

TEST(DiningPhilosophersMPI, Test_with_world_size_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  if (num_philosophers >= 2) {
    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(dining_task.validation());
}

TEST(DiningPhilosophersMPI, Test_with_0_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = 0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(DiningPhilosophersMPI, Test_with_1_philosopher) {
  boost::mpi::communicator world;

  int num_philosophers = 1;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(DiningPhilosophersMPI, Test_with_negative_size_philosophers) {
  boost::mpi::communicator world;

  int num_philosophers = -3;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

  ASSERT_FALSE(dining_task.validation());
}

TEST(DiningPhilosophersMPI, Test_with_2_philosophers) {
  boost::mpi::communicator world;
  if (world.size() >= 2) {
    int num_philosophers = 2;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(num_philosophers);

    beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(DiningPhilosophersMPI, Test_with_3_philosophers) {
  boost::mpi::communicator world;
  if (world.size() >= 3) {
    int num_philosophers = 3;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(num_philosophers);

    beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(DiningPhilosophersMPI, Test_with_4_philosophers) {
  boost::mpi::communicator world;
  if (world.size() >= 4) {
    int num_philosophers = 4;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(num_philosophers);

    beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(DiningPhilosophersMPI, Test_with_8_philosophers) {
  boost::mpi::communicator world;
  if (world.size() >= 8) {
    int num_philosophers = 8;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(num_philosophers);

    beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int> dining_task(taskData);

    ASSERT_TRUE(dining_task.validation());
    ASSERT_TRUE(dining_task.pre_processing());
    ASSERT_TRUE(dining_task.run());
    ASSERT_TRUE(dining_task.post_processing());
    bool deadlock_detected = dining_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}