// Copyright 2024 Koshkin Matvey
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>

#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

using Params = std::tuple<int, int>;

class koshkin_m_pholosophers_test : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(koshkin_m_pholosophers_test, test_task) {
  const auto& [f, s] = GetParam();

  boost::mpi::communicator world;
  std::vector<int> global_vec(f, s);
  std::vector<int32_t> average_value(1, 0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  koshkin_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_FALSE(testMpiTaskParallel.validation());
    }
  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(koshkin_m_pholosophers_test, koshkin_m_pholosophers_test,
                         ::testing::Values(Params(1, 1), Params(1, 2), Params(1, 3)));