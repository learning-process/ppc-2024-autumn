#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/prokhorov_n_producer_customer/include/ops_mpi.hpp"

TEST(prokhorov_n_producer_customer_mpi, Test_Sequence_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  size_t start = 2;
  size_t end = 6;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, (world.size() - 1) / 2);

  int num_producers = dis(gen);

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) global_vec.push_back(i + 1);
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(start);
    taskDataPar->inputs_count.emplace_back(end);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Doubled_Numbers_Processes) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  size_t start = 2;
  size_t end = 6;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, (world.size() - 1) / 2);

  int num_producers = dis(gen);

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      global_vec.push_back((i + 1) * 2);
    }
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(start);
    taskDataPar->inputs_count.emplace_back(end);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  prokhorov_n_producer_customer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < num_producers; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}
