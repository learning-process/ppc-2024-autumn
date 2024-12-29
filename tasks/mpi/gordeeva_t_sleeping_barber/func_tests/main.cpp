#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

TEST(gordeeva_t_sleeping_barber_mpi, Test_Validation1) {
  boost::mpi::communicator world;
  int i = 0;
  int j = 0;
  int k = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count = {0};
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Validation2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    if (world.size() < 3) {
      taskDataPar->inputs_count = {1};
      EXPECT_FALSE(testMpiTaskParallel.validation());
    }
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Validation3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(gordeeva_t_sleeping_barber_mpi, Test_End_To_End1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 1;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));
  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(gordeeva_t_sleeping_barber_mpi, Test_End_To_End2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 2;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(gordeeva_t_sleeping_barber_mpi, Test_End_To_End3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 3;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(gordeeva_t_sleeping_barber_mpi, Test_End_To_End4) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 996;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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

TEST(gordeeva_t_sleeping_barber_mpi, Test_End_To_End5) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs = 999;
  int global_res = -1;

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

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
