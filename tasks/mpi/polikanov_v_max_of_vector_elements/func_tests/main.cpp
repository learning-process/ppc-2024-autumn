// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/polikanov_v_max_of_vector_elements/include/ops_mpi.hpp"

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Valid_false) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(100, 1);
  std::vector<int32_t> global_sum(2, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = polikanov_v_max_of_vector_elements::getRandomVector(100, 0, 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  polikanov_v_max_of_vector_elements::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Valid_true) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(100, 1);
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  polikanov_v_max_of_vector_elements::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
}
TEST(polikanov_v_max_of_vector_elements_MPI, Test_Main) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  int n = 10000000;
  int lower = 0;
  int max_el = 100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = polikanov_v_max_of_vector_elements::getRandomVector(n, lower, max_el);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(max_el, ans[0]);
  }
}
