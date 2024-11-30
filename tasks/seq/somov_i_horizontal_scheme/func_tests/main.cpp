#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <random>
#include <vector>

#include "seq/somov_i_horizontal_scheme/include/ops_seq.hpp"

namespace somov_i_horizontal_scheme {

std::vector<std::vector<int32_t>> createRandomMatrix(uint32_t rowCount, uint32_t colCount) {
  std::random_device randomDevice;
  std::mt19937 generator(randomDevice());
  std::normal_distribution<float> distribution(0.0f, 100.0f);

  std::vector<std::vector<int32_t>> matrix(rowCount, std::vector<int32_t>(colCount));
  for (auto& row : matrix) {
    for (auto& element : row) {
      element = std::clamp(static_cast<int32_t>(std::round(distribution(generator))), -300, 300);
    }
  }
  return matrix;
}

std::vector<int32_t> createRandomVector(uint32_t vectorSize) {
  std::random_device randomDevice;
  std::mt19937 generator(randomDevice());
  std::normal_distribution<float> distribution(0.0f, 100.0f);

  std::vector<int32_t> vector(vectorSize);
  for (auto& element : vector) {
    element = std::clamp(static_cast<int32_t>(std::round(distribution(generator))), -300, 300);
  }
  return vector;
}

}  // namespace somov_i_horizontal_scheme

TEST(somov_i_horizontal_scheme, validateSquareMatrix) {
  uint32_t rowCount = 100, colCount = 100;
  auto matrix = somov_i_horizontal_scheme::createRandomMatrix(rowCount, colCount);
  auto vector = somov_i_horizontal_scheme::createRandomVector(colCount);
  std::vector<int32_t> result(rowCount);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : matrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(somov_i_horizontal_scheme, validateZeroValues) {
  uint32_t rowCount = 4, colCount = 4;
  std::vector<std::vector<int32_t>> zeroMatrix(rowCount, std::vector<int32_t>(colCount, 0));
  std::vector<int32_t> zeroVector(colCount, 0);
  std::vector<int32_t> result(rowCount, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : zeroMatrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(zeroVector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  for (uint32_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], 0);
  }
}

TEST(somov_i_horizontal_scheme, validateEmptyMatrix) {
  uint32_t rowCount = 0, colCount = 0;
  std::vector<std::vector<int32_t>> emptyMatrix(rowCount, std::vector<int32_t>(colCount));
  std::vector<int32_t> emptyVector(colCount, 0);
  std::vector<int32_t> result(rowCount, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : emptyMatrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(emptyVector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(somov_i_horizontal_scheme, validateUniformMatrix) {
  uint32_t rowCount = 5, colCount = 5;
  std::vector<std::vector<int32_t>> uniformMatrix(rowCount, std::vector<int32_t>(colCount, 42));
  std::vector<int32_t> uniformVector(colCount, 42);
  std::vector<int32_t> result(rowCount);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : uniformMatrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(uniformVector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(somov_i_horizontal_scheme, validateNonStandardValues) {
  uint32_t rowCount = 3, colCount = 3;
  std::vector<std::vector<int32_t>> nonStandardMatrix(rowCount, std::vector<int32_t>(colCount));
  nonStandardMatrix[0] = {INT_MAX, INT_MIN, 0};
  nonStandardMatrix[1] = {INT_MIN, INT_MAX, 100};
  nonStandardMatrix[2] = {200, -200, 0};
  std::vector<int32_t> nonStandardVector(colCount, 100);
  std::vector<int32_t> result(rowCount);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : nonStandardMatrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(nonStandardVector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(somov_i_horizontal_scheme, validationFailureTest) {
  uint32_t rowCount = 5, colCount = 10;
  auto matrix = somov_i_horizontal_scheme::createRandomMatrix(rowCount, colCount);
  auto vector = somov_i_horizontal_scheme::createRandomVector(colCount);
  std::vector<int32_t> result(rowCount);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  somov_i_horizontal_scheme::MatrixVectorTask task(taskData);

  for (auto& row : matrix) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskData->inputs_count = {rowCount, colCount};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(0);

  ASSERT_FALSE(task.validation());
}