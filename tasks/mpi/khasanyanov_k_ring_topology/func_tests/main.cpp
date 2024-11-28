#include "../include/tests.hpp"

RUN_FUNC_TESTS()

TEST(khasanyanov_k_ring_topology_tests, test_order) {
  const int size = 10;
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  std::vector<int> actual = khasanyanov_k_ring_topology_mpi::RingTopology<int>::true_order(size);
  ASSERT_EQ(expected, actual);
}

TEST(khasanyanov_k_ring_topology_tests, test_validation) {
  boost::mpi::communicator world;
  const std::vector<int> in_data = khasanyanov_k_ring_topology_mpi::generate_random_vector<int>(0);
  std::vector<int> out_data(in_data);
  std::vector<int> order(0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<int>(out_data, order);
  }
  khasanyanov_k_ring_topology_mpi::RingTopology<int> testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

TEST(khasanyanov_k_ring_topology_tests, test_creating_task_data) {
  boost::mpi::communicator world;
  const std::vector<int> in_data = khasanyanov_k_ring_topology_mpi::generate_random_vector<int>(10);
  std::vector<int> out_data(in_data);
  std::vector<int> order(world.size());
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_ring_topology_mpi::create_task_data<int>(out_data, order);
    ASSERT_FALSE(taskData->inputs.empty());
    ASSERT_FALSE(taskData->inputs_count.empty());
    EXPECT_TRUE(taskData->outputs.size() == 2);
    EXPECT_TRUE(taskData->outputs_count.size() == 2);
  }
}
