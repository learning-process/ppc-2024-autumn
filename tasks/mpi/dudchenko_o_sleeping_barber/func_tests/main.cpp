#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count = {0};
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    if (world.size() < 3) {
      taskDataPar->inputs_count = {1};
      EXPECT_FALSE(testMpiTaskParallel.validation());
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    if (world.size() < 3) {
      taskDataPar->inputs_count = {1};
      EXPECT_FALSE(testMpiTaskParallel.validation());
    } else {
      taskDataPar->inputs_count = {1};
      EXPECT_TRUE(testMpiTaskParallel.validation());
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 1;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    world.barrier();

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 3;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    world.barrier();

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 996;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    world.barrier();

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_4) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 999;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    world.barrier();
    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);
    }
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_5) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 1024;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber testMpiTaskParallel(taskDataPar);

  if (world.size() < 3) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    world.barrier();

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);
    }
  }
}