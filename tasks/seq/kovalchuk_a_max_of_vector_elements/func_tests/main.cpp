// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_5_5) {
  const int count_rows = 5;
  const int count_columns = 5;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(count_rows, count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();
  // Create data
  std::vector<int32_t> reference_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeqRef->inputs_count.emplace_back(count_rows);
  taskDataSeqRef->inputs_count.emplace_back(count_columns);
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTaskRef(taskDataSeqRef);
  ASSERT_EQ(testSequentialTaskRef.validation(), true);
  testSequentialTaskRef.pre_processing();
  testSequentialTaskRef.run();
  testSequentialTaskRef.post_processing();
  ASSERT_EQ(reference_max[0], global_max[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_10_10) {
  const int count_rows = 10;
  const int count_columns = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(count_rows, count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();
  // Create data
  std::vector<int32_t> reference_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeqRef->inputs_count.emplace_back(count_rows);
  taskDataSeqRef->inputs_count.emplace_back(count_columns);
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTaskRef(taskDataSeqRef);
  ASSERT_EQ(testSequentialTaskRef.validation(), true);
  testSequentialTaskRef.pre_processing();
  testSequentialTaskRef.run();
  testSequentialTaskRef.post_processing();
  ASSERT_EQ(reference_max[0], global_max[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_50_20) {
  const int count_rows = 50;
  const int count_columns = 20;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(count_rows, count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();
  // Create data
  std::vector<int32_t> reference_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeqRef->inputs_count.emplace_back(count_rows);
  taskDataSeqRef->inputs_count.emplace_back(count_columns);
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTaskRef(taskDataSeqRef);
  ASSERT_EQ(testSequentialTaskRef.validation(), true);
  testSequentialTaskRef.pre_processing();
  testSequentialTaskRef.run();
  testSequentialTaskRef.post_processing();
  ASSERT_EQ(reference_max[0], global_max[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_1_100) {
  const int count_rows = 1;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(count_rows, count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();
  // Create data
  std::vector<int32_t> reference_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeqRef->inputs_count.emplace_back(count_rows);
  taskDataSeqRef->inputs_count.emplace_back(count_columns);
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTaskRef(taskDataSeqRef);
  ASSERT_EQ(testSequentialTaskRef.validation(), true);
  testSequentialTaskRef.pre_processing();
  testSequentialTaskRef.run();
  testSequentialTaskRef.post_processing();
  ASSERT_EQ(reference_max[0], global_max[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_4_200) {
  const int count_rows = 4;
  const int count_columns = 200;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(count_rows, count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();
  // Create data
  std::vector<int32_t> reference_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeqRef->inputs_count.emplace_back(count_rows);
  taskDataSeqRef->inputs_count.emplace_back(count_columns);
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());
  // Create Task
  kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask testSequentialTaskRef(taskDataSeqRef);
  ASSERT_EQ(testSequentialTaskRef.validation(), true);
  testSequentialTaskRef.pre_processing();
  testSequentialTaskRef.run();
  testSequentialTaskRef.post_processing();
  ASSERT_EQ(reference_max[0], global_max[0]);
}