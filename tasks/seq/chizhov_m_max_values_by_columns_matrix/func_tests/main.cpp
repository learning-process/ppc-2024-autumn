// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/chizhov_m_max_values_by_columns_matrix/include/ops_seq.hpp"

TEST(chizhov_m_max_values_by_columns_matrix_seq, Test_Empty_Matrix) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int columns = 3;
  std::vector<int> matrix;
  std::vector<int> res_seq(columns, 0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&columns));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
  taskDataSeq->inputs_count.emplace_back(res_seq.size());

  chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(chizhov_m_max_values_by_columns_matrix_seq, Test_Max_3_Columns) {
  int columns = 3;

  // Create data
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> max(columns, 0);
  std::vector<int> result = {7, 8, 9};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&columns));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSeq->outputs_count.emplace_back(max.size());

  // Create Task
  chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(chizhov_m_max_values_by_columns_matrix_seq, Test_Max_4_Columns) {
  int columns = 4;

  // Create data
  std::vector<int> matrix = {4, 7, 5, 3, 8, 10, 12, 4, 2, 15, 3, 27};
  std::vector<int> max(columns, 0);
  std::vector<int> result = {8, 15, 12, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&columns));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSeq->outputs_count.emplace_back(max.size());

  // Create Task
  chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(chizhov_m_max_values_by_columns_matrix_seq, Test_Max_5_Columns) {
  int columns = 5;

  // Create data
  std::vector<int> matrix = {4, 7, 5, 3, 8, 10, 12, 4, 2, 6, 2, 1, 15, 3, 27};
  std::vector<int> max(columns, 0);
  std::vector<int> result = {10, 12, 15, 3, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&columns));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSeq->outputs_count.emplace_back(max.size());

  // Create Task
  chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(chizhov_m_max_values_by_columns_matrix_seq, Test_Max_6_Columns) {
  int columns = 6;

  // Create data
  std::vector<int> matrix = {9, 20, 3, 4, 7, 5, 3, 8, 10, 12, 4, 2, 6, 2, 1, 15, 3, 27};
  std::vector<int> max(columns, 0);
  std::vector<int> result = {9, 20, 10, 15, 7, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&columns));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSeq->outputs_count.emplace_back(max.size());

  // Create Task
  chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}