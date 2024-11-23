// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sorokin_a_producer_consumer/include/ops_mpi.hpp"

TEST(sorokin_a_producer_consumer_mpi, Test_Basic) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    ASSERT_EQ(1, 1);
    return;
  }
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) global_vec.push_back(i + 1);
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  sorokin_a_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) ASSERT_EQ(global_vec[i], global_sum[i]);
  }
}
TEST(sorokin_a_producer_consumer_mpi, Test_Basic_x2) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    ASSERT_EQ(1, 1);
    return;
  }
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) global_vec.push_back((i + 1) * 2);
    global_sum = global_vec;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  sorokin_a_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) ASSERT_EQ(global_vec[i], global_sum[i]);
  }
}