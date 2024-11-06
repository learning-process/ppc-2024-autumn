#include <gtest/gtest.h>

#include <vector>

#include "seq/varfolomeev_g_matrix_max_rows_vals/include/ops_seq.hpp"

std::vector<std::vector<int>> generateMatrix(int rows, int cols, int a, int b) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  // set generator
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = std::rand() % (b - a + 1) + a;
    }
  }
  return matrix;
}

int searchMaxInVec(std::vector<int> vec) {
  int max = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (max < vec[i]) max = vec[i];
  }
  return max;
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_in_empty) {
  int rows = 0;
  int cols = 0;

  // Create data
  std::vector<int> in;
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();
  ASSERT_EQ((int)out.size(), 0);
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_non_generated_4x4) {
  int rows = 4;
  int cols = 4;
  // Create data;
  std::vector<std::vector<int>> in = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  std::vector<int> expected_max = {4, 8, 12, 16};
  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(out[i], expected_max[i]);
  }
}

// Тест на матрицу с отрицательными значениями
TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_non_generated_negative_values) {
  int rows = 3;
  int cols = 3;

  // Create data
  std::vector<std::vector<int>> in = {{-10, -20, -30}, {-40, -50, -60}, {-70, -80, -90}};
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  std::vector<int> expected_max = {-10, -40, -70};
  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ(out[i], expected_max[i]);
  }
}

// Тест на матрицу с одинаковыми значениями
TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_same_values) {
  const int rows = 3;
  const int cols = 3;

  // Create data
  std::vector<std::vector<int>> in = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  std::vector<int> expected_max = {5, 5, 5};
  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ(out[i], expected_max[i]);
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generateMatrix_generator) {
  int rows = 5;
  int cols = 10;
  int a = -50;
  int b = 50;

  std::vector<std::vector<int>> matrix = generateMatrix(rows, cols, a, b);

  // Check size
  ASSERT_EQ((int)matrix.size(), rows);
  for (int i = 0; i < rows; ++i) {
    ASSERT_EQ((int)matrix[i].size(), cols);
  }

  // Check diap
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ASSERT_GE(matrix[i][j], a);
      ASSERT_LE(matrix[i][j], b);
    }
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_10x10) {
  int rows = 10;
  int cols = 10;

  // Create data; generation matrix integers from -100 to 100
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, -100, 100);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  for (int i = 0; i < rows; i++) {
    int expected_max = *std::max_element(in[i].begin(), in[i].end());
    ASSERT_EQ(out[i], expected_max);
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_20x10) {
  int rows = 20;
  int cols = 10;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, -100, 100);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = *std::max_element(in[i].begin(), in[i].end());
    ASSERT_EQ(out[i], expected_max);
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_10x20) {
  int rows = 10;
  int cols = 20;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, -100, 100);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = *std::max_element(in[i].begin(), in[i].end());
    ASSERT_EQ(out[i], expected_max);
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_20x20) {
  int rows = 20;
  int cols = 20;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, -100, 100);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = *std::max_element(in[i].begin(), in[i].end());
    ASSERT_EQ(out[i], expected_max);
  }
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_50x50) {
  int rows = 50;
  int cols = 50;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, -100, 100);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows maxInRows(taskDataSeq);
  ASSERT_EQ(maxInRows.validation(), true);
  maxInRows.pre_processing();
  maxInRows.run();
  maxInRows.post_processing();

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = *std::max_element(in[i].begin(), in[i].end());
    ASSERT_EQ(out[i], expected_max);
  }
}
