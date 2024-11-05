#include <gtest/gtest.h>

#include <vector>

#include "seq/varfolomeev_g_matrix_max_rows_vals/include/ops_seq.hpp"

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_in_empty) {
  const int rows = 0, cols = 0;

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
  ASSERT_EQ(out.size(), 0);
}

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_non_generated_4x4) {
  const int rows = 4, cols = 4;

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
  const int rows = 3;
  const int cols = 3;

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

TEST(varfolomeev_g_matrix_max_rows_vals_seq, Test_generated_10x10) {
  const int rows = 10, cols = 10;

  // Create data; generation matrix integers from -100 to 100
  std::vector<std::vector<int>> in = varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(rows, cols, -100, 100);
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
  const int rows = 20, cols = 10;

  // Create data
  std::vector<std::vector<int>> in = varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(rows, cols, -100, 100);
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
  const int rows = 10, cols = 20;

  // Create data
  std::vector<std::vector<int>> in = varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(rows, cols, -100, 100);
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
  const int rows = 20, cols = 20;

  // Create data
  std::vector<std::vector<int>> in = varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(rows, cols, -100, 100);
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
  const int rows = 50, cols = 50;

  // Create data
  std::vector<std::vector<int>> in = varfolomeev_g_matrix_max_rows_vals_seq::generateMatrix(rows, cols, -100, 100);
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
