#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_Zero_Value) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 0;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
    taskDataPar->inputs_count = {size};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_parallel.data()));
    taskDataPar->outputs_count = {size};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_FALSE(testPar.validation());
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_Empty_Output) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  const int size = 0;

  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
    taskDataPar->inputs_count = {size};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_FALSE(testPar.validation());
 }

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_100_with_random) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 100;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
    taskDataPar->inputs_count = {static_cast<size_t>(size)};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_parallel.data()));
    taskDataPar->outputs_count = {static_cast<size_t>(size)};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_TRUE(testPar.validation());

  ASSERT_TRUE(testPar.pre_processing());
  ASSERT_TRUE(testPar.run());
  ASSERT_TRUE(testPar.post_processing());

  world.barrier();

  if (world.rank() == 0) {
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_EQ(result_parallel, expected_vec);
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_1000_with_random) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 1000;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
    taskDataPar->inputs_count = {static_cast<size_t>(size)};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_parallel.data()));
    taskDataPar->outputs_count = {static_cast<size_t>(size)};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_TRUE(testPar.validation());

  ASSERT_TRUE(testPar.pre_processing());
  ASSERT_TRUE(testPar.run());
  ASSERT_TRUE(testPar.post_processing());

  world.barrier();

  if (world.rank() == 0) {
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_EQ(result_parallel, expected_vec);
  }
}

TEST(gordeeva_t_shell_sort_batcher_merge_mpi, Shell_sort_5000_with_random) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int size = 5000;
  std::vector<int> input_vec;
  std::vector<int> result_parallel(size);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_vec = gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::rand_vec(size, 0, 1000);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
    taskDataPar->inputs_count = {size};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_parallel.data()));
    taskDataPar->outputs_count = {size};
  }

  gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel testPar(taskDataPar);

  ASSERT_TRUE(testPar.validation());
  testPar.pre_processing();
  testPar.run();
  testPar.post_processing();

  world.barrier();

  if (world.rank() == 0) {
    auto expected_vec = input_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_EQ(result_parallel, expected_vec);
  }
}
