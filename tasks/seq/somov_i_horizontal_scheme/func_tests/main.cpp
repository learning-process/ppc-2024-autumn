#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <random>
#include <vector>

#include "seq/somov_i_horizontal_scheme/include/ops_seq.hpp"

namespace somov_i_horizontal_scheme {

std::vector<int32_t> create_random_matrix(uint32_t rowCount, uint32_t colCount) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> matrix(rowCount * colCount);
  for (auto &el : matrix) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return matrix;
}

std::vector<int32_t> create_random_vector(uint32_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> vec(size);
  for (auto &el : vec) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return vec;
}

}  // namespace somov_i_horizontal_scheme

TEST(somov_i_horizontal_scheme, validate_random_matrix) {
  uint32_t rowCount = 600;
  uint32_t colCount = 600;

  auto matrix = somov_i_horizontal_scheme::create_random_matrix(rowCount, colCount);
  auto vector = somov_i_horizontal_scheme::create_random_vector(colCount);
  std::vector<int32_t> result(rowCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_non_standard_values) {
  uint32_t rowCount = 10;
  uint32_t colCount = 10;

  std::vector<int32_t> non_standard_matrix = {INT_MAX, INT_MIN, 0, INT_MIN, INT_MAX, 100, 200, -200, 0};
  non_standard_matrix.resize(rowCount * colCount, 0);
  std::vector<int32_t> non_standard_vector = {100, 100, 100};
  non_standard_vector.resize(colCount, 0);
  std::vector<int32_t> result(rowCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(non_standard_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(non_standard_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_zero_matrix) {
  uint32_t rowCount = 7;
  uint32_t colCount = 7;

  std::vector<int32_t> zero_matrix(rowCount * colCount, 0);
  std::vector<int32_t> zero_vector(colCount, 0);
  std::vector<int32_t> result(rowCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(zero_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(zero_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_empty_matrix) {
  uint32_t rowCount = 0;
  uint32_t colCount = 0;

  std::vector<int32_t> empty_matrix;
  std::vector<int32_t> empty_vector;
  std::vector<int32_t> result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(empty_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(empty_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_uniform_matrix) {
  uint32_t rowCount = 81;
  uint32_t colCount = 17;

  std::vector<int32_t> uniform_matrix(rowCount * colCount, 42);
  std::vector<int32_t> uniform_vector(colCount, 42);
  std::vector<int32_t> result(rowCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_uniform_matrix_two) {
  uint32_t rowCount = 64;
  uint32_t colCount = 100;

  std::vector<int32_t> uniform_matrix(rowCount * colCount, 37);
  std::vector<int32_t> uniform_vector(colCount, 37);
  std::vector<int32_t> result(rowCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}

TEST(somov_i_horizontal_scheme, validate_uniform_matrix_three) {
  uint32_t rowCount = 17;
  uint32_t colCount = 32;

  std::vector<int32_t> uniform_matrix(rowCount * colCount, 22);
  std::vector<int32_t> uniform_vector(colCount, 22);
  std::vector<int32_t> result(rowCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_matrix.data()));
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(uniform_vector.data()));
  task_data->inputs_count = {rowCount, colCount};
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.push_back(rowCount);

  auto seq_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(task_data);
  seq_task->setRowCount(rowCount);
  seq_task->setColCount(colCount);

  ASSERT_TRUE(seq_task->validation());
  seq_task->pre_processing();
  seq_task->run();
  seq_task->post_processing();
}