#include <boost/mpi.hpp>
#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <thread>

#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

namespace kazunin_n_dining_philosophers {

TEST(KazuninDiningPhilosophersMPI, ValidInputTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count < 2) {
    GTEST_SKIP() << "Test skipped: Not enough philosophers to simulate.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for valid input.";

  philosophers_task->pre_processing();
  ASSERT_NO_THROW(philosophers_task->run()) << "Exception thrown during valid execution.";
  philosophers_task->post_processing();

  mpi_comm.barrier(); 
}

TEST(KazuninDiningPhilosophersMPI, EatingThinkingSimulationTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count < 2) {
    GTEST_SKIP() << "Test skipped: Not enough philosophers to simulate eating and thinking.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for eating and thinking simulation.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  ASSERT_EQ(philosophers_task->state, PhilosophersState::THINKING) << "Philosopher should be thinking.";

  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();
  ASSERT_EQ(philosophers_task->state, PhilosophersState::EATING) << "Philosopher should be eating.";
  philosophers_task->post_processing();

  mpi_comm.barrier(); 
}

TEST(KazuninDiningPhilosophersMPI, DeadlockSimulationTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count < 3) {
    GTEST_SKIP() << "Test skipped: Not enough philosophers to simulate a potential deadlock.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for deadlock simulation.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();

  bool deadlock_resolved = false;
  for (int i = 0; i < 100; ++i) {
    if (!philosophers_task->detect_deadlock()) {
      deadlock_resolved = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(deadlock_resolved) << "Deadlock was not resolved as expected.";
  philosophers_task->post_processing();

  mpi_comm.barrier(); 
}

TEST(KazuninDiningPhilosophersMPI, TaskCompletionTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count < 2) {
    GTEST_SKIP() << "Test skipped: Not enough philosophers to simulate.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for task completion.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();

  for (int i = 0; i < 5; ++i) {
    philosophers_task->simulate_thinking();
    philosophers_task->request_forks_from_neighbors();
    philosophers_task->simulate_eating();
  }

  bool task_completed = false;
  for (int i = 0; i < 100; ++i) {
    if (philosophers_task->check_termination()) {
      task_completed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(task_completed) << "Task did not complete as expected.";
  philosophers_task->post_processing();

  mpi_comm.barrier(); 
}

TEST(KazuninDiningPhilosophersMPI, DeadlockResolutionTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count < 3) {
    GTEST_SKIP() << "Test skipped: Not enough philosophers to simulate and resolve deadlock.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for deadlock resolution.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();

  bool deadlock_resolved = false;
  for (int i = 0; i < 100; ++i) {
    if (philosophers_task->detect_deadlock()) {
      philosophers_task->resolve_deadlock();
    }
    if (!philosophers_task->detect_deadlock()) {
      deadlock_resolved = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(deadlock_resolved) << "Deadlock was not resolved as expected.";
  philosophers_task->post_processing();

  mpi_comm.barrier(); 
}

TEST(KazuninDiningPhilosophersMPI, ThreePhilosophersTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count != 3) {
    GTEST_SKIP() << "Test skipped: This test requires exactly 3 philosophers.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for 3 philosophers.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();
  philosophers_task->post_processing();
}

TEST(KazuninDiningPhilosophersMPI, FourPhilosophersTest) {
  boost::mpi::communicator mpi_comm;
  int philosophers_count = mpi_comm.size();
  if (philosophers_count != 4) {
    GTEST_SKIP() << "Test skipped: This test requires exactly 4 philosophers.";
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(philosophers_count);

  auto philosophers_task = std::make_shared<KazuninDiningPhilosophersMPI<int>>(task_data);
  ASSERT_TRUE(philosophers_task->validation()) << "Validation failed for 4 philosophers.";

  philosophers_task->pre_processing();
  philosophers_task->simulate_thinking();
  philosophers_task->request_forks_from_neighbors();
  philosophers_task->simulate_eating();
  philosophers_task->post_processing();
}

}  // namespace kazunin_n_dining_philosophers
