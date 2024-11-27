#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>

#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"
#include "mpi/kazunin_n_dining_philosophers/src/ops_mpi.cpp"

TEST(KazuninDiningPhilosophersMPI, TestWith1Philosopher) {
  boost::mpi::communicator world;

  int count_philosophers = 1;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  ASSERT_FALSE(philosopher_task.validation());
}

TEST(KazuninDiningPhilosophersMPI, TestWith2Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 2;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith3Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 3;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith4Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 4;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith5Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 5;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith6Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 6;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith7Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 7;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWith8Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 8;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (philosopher_task.validation()) {
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP();
  }
}

TEST(KazuninDiningPhilosophersMPI, TestWithWorldSizePhilosophers) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  if (count_philosophers >= 2) {
    ASSERT_TRUE(philosopher_task.validation());
    ASSERT_TRUE(philosopher_task.pre_processing());
    ASSERT_TRUE(philosopher_task.run());
    ASSERT_TRUE(philosopher_task.post_processing());
    bool deadlock_detected = philosopher_task.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(philosopher_task.validation());
}

TEST(KazuninDiningPhilosophersMPI, TestWithNegativeSizePhilosophers) {
  boost::mpi::communicator world;

  int count_philosophers = -5;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int> philosopher_task(taskData);

  ASSERT_FALSE(philosopher_task.validation());
}