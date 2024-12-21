#include <gtest/gtest.h>

#include <vector>

#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Zero_Columns) {
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  int numCols = 0;
  std::vector<int> matrix;
  std::vector<int> resultSequential(numCols, 0);

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(resultSequential.data()));
  taskDataSequential->inputs_count.emplace_back(resultSequential.size());

  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Empty_Matrix) {
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  int numCols = 3;
  std::vector<int> matrix;
  std::vector<int> resultSequential(numCols, 0);

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(resultSequential.data()));
  taskDataSequential->inputs_count.emplace_back(resultSequential.size());

  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Max_3_Columns) {
  int numCols = 3;

  // Create data
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> max(numCols, 0);
  std::vector<int> result = {7, 8, 9};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSequential->outputs_count.emplace_back(max.size());

  // Create Task
  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Max_4_Columns) {
  int numCols = 4;

  // Create data
  std::vector<int> matrix = {4, 7, 5, 3, 8, 10, 12, 4, 2, 15, 3, 27};
  std::vector<int> max(numCols, 0);
  std::vector<int> result = {8, 15, 12, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSequential->outputs_count.emplace_back(max.size());

  // Create Task
  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Max_5_Columns) {
  int numCols = 5;

  // Create data
  std::vector<int> matrix = {4, 7, 5, 3, 8, 10, 12, 4, 2, 6, 2, 1, 15, 3, 27};
  std::vector<int> max(numCols, 0);
  std::vector<int> result = {10, 12, 15, 3, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSequential->outputs_count.emplace_back(max.size());

  // Create Task
  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}

TEST(nasedkin_e_matrix_column_max_value_seq, Test_Max_6_Columns) {
  int numCols = 6;

  // Create data
  std::vector<int> matrix = {9, 20, 3, 4, 7, 5, 3, 8, 10, 12, 4, 2, 6, 2, 1, 15, 3, 27};
  std::vector<int> max(numCols, 0);
  std::vector<int> result = {9, 20, 10, 15, 7, 27};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSequential->inputs_count.emplace_back(matrix.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&numCols));
  taskDataSequential->inputs_count.emplace_back((size_t)1);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(max.data()));
  taskDataSequential->outputs_count.emplace_back(max.size());

  // Create Task
  nasedkin_e_matrix_column_max_value_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, max);
}