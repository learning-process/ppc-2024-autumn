// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"
#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

TEST(kovalchuk_a_max_of_vector_elements, Test_Max_5_5) {
  const int count_rows = 5;
  const int count_columns = 5;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    global_matrix = kovalchuk_a_max_of_vector_elements::getRandomMatrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  kovalchuk_a_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, INT_MIN);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
    ASSERT_EQ(testSequentialTask.validation(), true);
    testSequentialTask.pre_processing();
    testSequentialTask.run();
    testSequentialTask.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements, Test_Max_10_10) {
  const int count_rows = 10;
  const int count_columns = 10;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    global_matrix = kovalchuk_a_max_of_vector_elements::getRandomMatrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  kovalchuk_a_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, INT_MIN);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
    ASSERT_EQ(testSequentialTask.validation(), true);
    testSequentialTask.pre_processing();
    testSequentialTask.run();
    testSequentialTask.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements, Test_Max_50_20) {
  const int count_rows = 50;
  const int count_columns = 20;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    global_matrix = kovalchuk_a_max_of_vector_elements::getRandomMatrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  kovalchuk_a_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, INT_MIN);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
    ASSERT_EQ(testSequentialTask.validation(), true);
    testSequentialTask.pre_processing();
    testSequentialTask.run();
    testSequentialTask.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements, Test_Max_100_100) {
  const int count_rows = 100;
  const int count_columns = 100;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    global_matrix = kovalchuk_a_max_of_vector_elements::getRandomMatrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  kovalchuk_a_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, INT_MIN);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
    ASSERT_EQ(testSequentialTask.validation(), true);
    testSequentialTask.pre_processing();
    testSequentialTask.run();
    testSequentialTask.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements, Test_Max_1_100) {
  const int count_rows = 1;
  const int count_columns = 100;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    global_matrix = kovalchuk_a_max_of_vector_elements::getRandomMatrix(count_rows, count_columns);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  kovalchuk_a_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, INT_MIN);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
    ASSERT_EQ(testSequentialTask.validation(), true);
    testSequentialTask.pre_processing();
    testSequentialTask.run();
    testSequentialTask.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}