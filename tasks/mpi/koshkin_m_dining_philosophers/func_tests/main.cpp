// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

TEST(koshkin_m_dining_philosophers, Test_Validation_Any_Process_Count) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Validation_Not_Enough_Forks) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> forks(world.size() - 1, 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Validation_Negative_Phil_Count) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int negative_count = -1;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&negative_count));
    taskDataPar->inputs_count.emplace_back(1);
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Validation_Empty_Data) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Validation_Invalid_Fork_Data) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::vector<int> invalid_forks(world.size() + 1, 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(invalid_forks.data()));
    taskDataPar->inputs_count.emplace_back(invalid_forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Validation_Valid_Data) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int local_num_philosophers = world.size();
    std::vector<int> forks(local_num_philosophers, world.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(koshkin_m_dining_philosophers, Test_Main_Simulation1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_philosophers = world.size();

  if (world.rank() == 0) {
    if (num_philosophers == 1) {
      std::vector<int> forks(num_philosophers, 2);
    }
  }

  broadcast(world, num_philosophers, 0);

  ASSERT_GT(num_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(num_philosophers, world.size());
  int iterations = 10;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(koshkin_m_dining_philosophers, Test_Main_Simulation2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_philosophers = world.size();

  if (world.rank() == 0) {
    if (num_philosophers == 1) {
      std::vector<int> forks(num_philosophers, 2);
    }
  }

  broadcast(world, num_philosophers, 0);

  ASSERT_GT(num_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(num_philosophers, world.size());
  int iterations = 100;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(koshkin_m_dining_philosophers, Test_Main_Simulation3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_philosophers = world.size();

  if (world.rank() == 0) {
    if (num_philosophers == 1) {
      std::vector<int> forks(num_philosophers, 2);
    }
  }

  broadcast(world, num_philosophers, 0);

  ASSERT_GT(num_philosophers, 0) << "Error: num_philosophers is not positive after broadcast";

  std::vector<int> forks(num_philosophers, world.size());
  int iterations = 300;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(koshkin_m_dining_philosophers, Test_Forks_State_After_Simulation) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_philosophers = world.size();
  std::vector<int> forks(num_philosophers, 2);

  if (world.rank() == 0) {
    int iterations = 100;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> final_forks(forks.size());
    memcpy(final_forks.data(), taskDataPar->inputs[0], forks.size() * sizeof(int));

    for (const auto& fork : final_forks) {
      ASSERT_EQ(fork, 2);
    }
  }
}

TEST(koshkin_m_dining_philosophers, Test_No_Hungry_Philosopher) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_philosophers = world.size();
  if (world.rank() == 0) {
    std::vector<int> forks(num_philosophers, 2);
    int iterations = 5;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(forks.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    taskDataPar->inputs_count.emplace_back(forks.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> eat_counts(world.size());
    memcpy(taskDataPar->inputs[0], eat_counts.data(), world.size() * sizeof(int));
    for (const auto& count : eat_counts) {
      ASSERT_EQ(count, 0);
    }
  }
}
