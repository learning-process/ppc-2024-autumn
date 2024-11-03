#include <gtest/gtest.h>

#include <vector>

#include "seq/morozov_e_min_val_in_rows_matrix/include/ops_seq.hpp"
std::vector<std::vector<int>> getRandomMatrix(int n, int m) {
  int left = 0;
  int right = 10005;

  // Создаем матрицу
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

  // Заполняем матрицу случайными значениями
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = left + std::rand() % (right - left + 1);
    }
  }
  for (int i = 0; i < n; ++i) {
    int m_ = std::rand() % m;
    matrix[i][m_] = -1;
  }
  return matrix;
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Validation_False0) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Validation_False1) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  for (size_t i = 0; i < matrix.size(); ++i) taskDataSeq->inputs_count.emplace_back(1);
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Validation_False2) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  for (size_t i = 0; i < matrix.size(); ++i)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}

TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Validation_True) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  std::vector<int> res = {1, 2};
  for (size_t i = 0; i < matrix.size(); ++i)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(matrix[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Main0) {
  std::vector<std::vector<int>> matrix;
  const int n = 2;
  const int m = 2;
  std::vector<int> resSeq(n);
  std::vector<int> res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  matrix = {{1, 2}, {3, 4}};
  for (size_t i = 0; i < matrix.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  // ASSERT_EQ(v, res2);
  ASSERT_EQ(resSeq[0], 1);
  ASSERT_EQ(resSeq[1], 3);
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Main1) {
  std::vector<std::vector<int>> matrix;
  const int n = 10;
  const int m = 10;
  std::vector<int> resSeq(n);
  std::vector<int> res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  matrix = getRandomMatrix(n, m);
  for (size_t i = 0; i < matrix.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  // ASSERT_EQ(v, res2);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(resSeq[i], -1);
  }
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Main2) {
  std::vector<std::vector<int>> matrix;
  const int n = 1000;
  const int m = 1000;
  std::vector<int> resSeq(n);
  std::vector<int> res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  matrix = getRandomMatrix(n, m);
  for (size_t i = 0; i < matrix.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  // ASSERT_EQ(v, res2);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(resSeq[i], -1);
  }
}
TEST(morozov_e_min_val_in_rows_matrix_Sequential, Test_Main3) {
  std::vector<std::vector<int>> matrix;
  const int n = 5000;
  const int m = 5000;
  std::vector<int> resSeq(n);
  std::vector<int> res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  matrix = getRandomMatrix(n, m);
  for (size_t i = 0; i < matrix.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  // ASSERT_EQ(v, res2);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(resSeq[i], -1);
  }
}