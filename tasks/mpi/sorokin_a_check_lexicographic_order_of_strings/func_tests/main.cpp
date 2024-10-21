// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sorokin_a_check_lexicographic_order_of_strings/include/ops_mpi.hpp"

TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, The_difference_is_in_1_characters) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'a', 'p', 'p', 'p'}, {'b', 'a', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataSeq->inputs_count.emplace_back(strs.size());
    taskDataSeq->inputs_count.emplace_back(strs[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}

TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, The_difference_is_in_1_characters_res1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'c', 'p', 'p', 'p'}, {'b', 'a', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataSeq->inputs_count.emplace_back(strs.size());
    taskDataSeq->inputs_count.emplace_back(strs[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, The_difference_is_in_3_characters_res1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'a', 'a', 'p', 'p'}, {'a', 'a', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataSeq->inputs_count.emplace_back(strs.size());
    taskDataSeq->inputs_count.emplace_back(strs[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sorokin_a_check_lexicographic_order_of_strings_mpi, The_difference_is_in_4_characters) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> strs = {{'a', 'p', 'p', 'a'}, {'a', 'p', 'p', 'p'}};
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataPar->inputs_count.emplace_back(strs.size());
    taskDataPar->inputs_count.emplace_back(strs[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < strs.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(strs[i].data()));
    taskDataSeq->inputs_count.emplace_back(strs.size());
    taskDataSeq->inputs_count.emplace_back(strs[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}
