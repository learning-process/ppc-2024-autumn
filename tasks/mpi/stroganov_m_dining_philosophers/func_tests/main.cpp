// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <thread>
#include <vector>

#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"

TEST(stroganov_m_dining_philosophers, Valid_Number_Of_Philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  testMpiTaskParallel.run();
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Default_Number_Of_Philosophers) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  testMpiTaskParallel.run();
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Concurrent_Access) {
  boost::mpi::communicator world;

  int count_philosophers = 4;

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  testMpiTaskParallel.run();
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    ASSERT_GE(count_philosophers, 2);
  }
}

TEST(stroganov_m_dining_philosophers, Custom_Run_Logic) {
  boost::mpi::communicator world;

  int count_philosophers = 3;
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Validation_Check) {
  boost::mpi::communicator world;

  {
    std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
    if (world.size() == 1) {
      GTEST_SKIP() << "Skipping test as world size = 1";
    }
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  {
    int count_philosophers = 5;
    std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
      taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
    }
    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  {
    int count_philosophers = 1;
    std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
      taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
    }
    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  {
    std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataMpi->inputs.clear();
      taskDataMpi->inputs_count.clear();
    }
    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  {
    std::vector<uint8_t> invalid_data(2);
    std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskDataMpi->inputs.emplace_back(invalid_data.data());
      taskDataMpi->inputs_count.emplace_back(invalid_data.size());
    }
    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}

TEST(stroganov_m_dining_philosophers, Deadlock_Free_Execution) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  ASSERT_TRUE(testMpiTaskParallel.run());

  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Initial_Forks_State) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  for (int i = 0; i < count_philosophers; ++i) {
    ASSERT_FALSE(testMpiTaskParallel.get_forks()[i]) << "Fork " << i << " should be free initially.";
  }
}

TEST(stroganov_m_dining_philosophers, Forks_Locking) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  int philosopher_id = world.rank();
  ASSERT_TRUE(testMpiTaskParallel.distribution_forks(philosopher_id));
  ASSERT_TRUE(testMpiTaskParallel.get_forks()[philosopher_id]) << "Left fork should be locked.";
  ASSERT_TRUE(testMpiTaskParallel.get_forks()[(philosopher_id + 1) % count_philosophers])
      << "Right fork should be locked.";
  testMpiTaskParallel.release_forks(philosopher_id);
  ASSERT_FALSE(testMpiTaskParallel.get_forks()[philosopher_id]) << "Left fork should be released.";
  ASSERT_FALSE(testMpiTaskParallel.get_forks()[(philosopher_id + 1) % count_philosophers])
      << "Right fork should be released.";
}

TEST(stroganov_m_dining_philosophers, Forks_Cannot_Used_When_Locked) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  int philosopher_id = world.rank();
  ASSERT_TRUE(testMpiTaskParallel.distribution_forks(philosopher_id));
  int neighbor_id = (philosopher_id + 1) % count_philosophers;
  std::thread neighbor_thread([&]() {
    ASSERT_FALSE(testMpiTaskParallel.distribution_forks(neighbor_id))
        << "Neighbor should not be able to acquire locked forks.";
  });
  neighbor_thread.join();
  testMpiTaskParallel.release_forks(philosopher_id);
}

TEST(stroganov_m_dining_philosophers, Forks_Released_After_Completion) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  for (int i = 0; i < count_philosophers; ++i) {
    ASSERT_FALSE(testMpiTaskParallel.get_forks()[i]) << "Fork " << i << " should be free after completion.";
  }
}

TEST(stroganov_m_dining_philosophers, Test_Iterations_10) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_philosophers = world.size();

  if (world.rank() == 0) {
    if (count_philosophers == 1) {
      std::vector<int> forks(count_philosophers, 2);
    }
  }

  broadcast(world, count_philosophers, 0);

  ASSERT_GT(count_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(count_philosophers, world.size());
  int iterations = 10;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int expected_philosophers_finished = world.size();
    int actual_philosophers_finished = *reinterpret_cast<int*>(taskDataPar->inputs[0]);
    ASSERT_EQ(expected_philosophers_finished, actual_philosophers_finished);
  }
}

TEST(stroganov_m_dining_philosophers, Test_Iterations_100) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_philosophers = world.size();

  if (world.rank() == 0) {
    if (count_philosophers == 1) {
      std::vector<int> forks(count_philosophers, 2);
    }
  }

  broadcast(world, count_philosophers, 0);

  ASSERT_GT(count_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(count_philosophers, world.size());
  int iterations = 100;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int expected_philosophers_finished = world.size();
    int actual_philosophers_finished = *reinterpret_cast<int*>(taskDataPar->inputs[0]);
    ASSERT_EQ(expected_philosophers_finished, actual_philosophers_finished);
  }
}

TEST(stroganov_m_dining_philosophers, Test_Iterations_500) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int count_philosophers = world.size();

  if (world.rank() == 0) {
    if (count_philosophers == 1) {
      std::vector<int> forks(count_philosophers, 2);
    }
  }

  broadcast(world, count_philosophers, 0);

  ASSERT_GT(count_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(count_philosophers, world.size());
  int iterations = 300;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int expected_philosophers_finished = world.size();
    int actual_philosophers_finished = *reinterpret_cast<int*>(taskDataPar->inputs[0]);
    ASSERT_EQ(expected_philosophers_finished, actual_philosophers_finished);
  }
}

TEST(stroganov_m_dining_philosophers, Simulation_Time_Check) {
  boost::mpi::communicator world;

  int count_philosophers = 4;
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  auto start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) {
    testMpiTaskParallel.run();
  }
  auto end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_time = end_time - start_time;
  ASSERT_LT(elapsed_time.count(), 2.0);  // Проверка, что выполнение занимает менее 2 секунд.

  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}