// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/zaytsev_bitwise_sort_evenodd_Batcher/include/ops_mpi.hpp"

TEST(zaytsev_bitwise_sort_evenodd_Batcher, CorrectSorting) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {4, -2, 7, -5, 3, 8, -1, 0, 6, -9, -10, 9};
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, AlreadySorted) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {-9, -5, -2, -1, 0, 3, 4, 6, 7, 8};
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}