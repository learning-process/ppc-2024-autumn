#include "../include/tests.hpp"
#include "boost/mpi/communicator.hpp"

TEST(khasanyanov_k_ring_topology_tests, test) {
  boost::mpi::communicator world;

  const std::vector<double> in_data = khasanyanov_k_ring_topology_mpi::generate_random_vector<double>(100);
  std::vector<double> out_data(in_data);
  std::vector<int> order(world.size());

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<double>(out_data, order);
  }
  khasanyanov_k_ring_topology_mpi::RingTopology<double> testTask(taskData);
  RUN_TASK(testTask);

  if (world.rank() == 0) {
    auto pattern_order = khasanyanov_k_ring_topology_mpi::RingTopology<int>::true_order(world.size());
    ASSERT_EQ(pattern_order, order);
    ASSERT_EQ(in_data, out_data);
  }
}

RUN_FUNC_TESTS()