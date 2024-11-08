#include <memory>

#include "../include/tests.hpp"
#include "boost/mpi/communicator.hpp"
#include "gtest/gtest.h"

TEST(khasanyanov_k_ring_topology_tests, test) {
  boost::mpi::communicator world;

  std::vector<int> in_data = {6, 542, 45, 346, 456, 4};
  std::vector<int> out_data(in_data.size());
  std::vector<int> out;
  auto out_ptr = std::make_shared<std::vector<int>>(out);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<int, size_t>(in_data);
  }
  khasanyanov_k_ring_topology_mpi::RingTopology<int> testTask(taskData, out_ptr);
  RUN_TASK(testTask);

  if (world.rank() == 0) {
    auto pattern_out = khasanyanov_k_ring_topology_mpi::RingTopology<int>::true_order(world.size());
    ASSERT_EQ(pattern_out, out);
    ASSERT_EQ(in_data, out_data);
  }

  SUCCEED();
}