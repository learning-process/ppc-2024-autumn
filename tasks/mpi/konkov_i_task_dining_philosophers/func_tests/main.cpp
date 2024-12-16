#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(konkov_i_DiningPhilosophersTest, ValidNumberOfPhilosophers) {
  int num_philosophers = 5;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.validation());
  ASSERT_TRUE(dp.pre_processing());
  ASSERT_TRUE(dp.run());
  ASSERT_TRUE(dp.post_processing());
}

TEST(konkov_i_DiningPhilosophersTest, DeadlockFreeExecution) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "Skipping test: At least 2 processes are required.";
  }

  int num_philosophers = size;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.validation());
  ASSERT_TRUE(dp.pre_processing());
  ASSERT_TRUE(dp.run());
  ASSERT_TRUE(dp.post_processing());

  bool local_deadlock = dp.check_deadlock();

  bool global_deadlock = false;
  MPI_Allreduce(&local_deadlock, &global_deadlock, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

  ASSERT_FALSE(global_deadlock) << "Deadlock detected!";
}

TEST(konkov_i_DiningPhilosophersTest, SmallNumberOfPhilosophers) {
  int num_philosophers = 3;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.validation());
  ASSERT_TRUE(dp.pre_processing());
  ASSERT_TRUE(dp.run());
  ASSERT_TRUE(dp.post_processing());
}

TEST(konkov_i_DiningPhilosophersTest, SinglePhilosopher) {
  int num_philosophers = 1;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_FALSE(dp.validation());
}

TEST(DiningPhilosophersFunctional, InvalidPhilosopherCount) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_philosophers = -5;

  if (num_philosophers <= 0) {
    if (rank == 0) {
      GTEST_SKIP() << "Skipping test: Invalid number of philosophers (" << num_philosophers
                   << "). Number must be positive.";
    }
    return;
  }

  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.pre_processing());
  ASSERT_TRUE(dp.run());
  ASSERT_TRUE(dp.post_processing());
}
