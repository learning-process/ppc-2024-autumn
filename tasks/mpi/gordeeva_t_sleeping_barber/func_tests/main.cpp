#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_0_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 0;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_3_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 3;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_15_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 15;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_50_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 50;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_100_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 100;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_500_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 500;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_, barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ(global_res[0], 1);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_Without_Seats_Client) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  // const int max_waiting_chairs_ = 3;
  bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {barber_busy_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, Test_Barber_Without_Barber_Busy) {
  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int max_waiting_chairs_ = 3;
  // bool barber_busy_ = false;
  std::vector<int32_t> global_res(1, 0);

  gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs = {max_waiting_chairs_};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
