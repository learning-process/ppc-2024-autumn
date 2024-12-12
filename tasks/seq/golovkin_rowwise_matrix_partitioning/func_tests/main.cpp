// Golovkin Maksim Task#2

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_rowwise_matrix_partitioning/include/ops_seq.hpp"
using namespace golovkin_rowwise_matrix_partitioning;
using ppc::core::TaskData;

TEST(golovkin_rowwise_matrix_partitioning, Multiply_SquareMatrices) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  std::vector<std::vector<double>> B = {{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}};
  std::vector<std::vector<double>> result(3, std::vector<double>(3, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  MatrixMultiplicationTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<std::vector<double>> expected = {{30.0, 24.0, 18.0}, {84.0, 69.0, 54.0}, {138.0, 114.0, 90.0}};
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(golovkin_rowwise_matrix_partitioning, Multiply_IdentityMatrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  std::vector<std::vector<double>> B = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  std::vector<std::vector<double>> result(3, std::vector<double>(3, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  MatrixMultiplicationTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<std::vector<double>> expected = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(golovkin_rowwise_matrix_partitioning, Multiply_RectangularMatrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  std::vector<std::vector<double>> B = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  MatrixMultiplicationTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<std::vector<double>> expected = {{58.0, 64.0}, {139.0, 154.0}};
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(golovkin_rowwise_matrix_partitioning, Multiply_EmptyMatrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {};
  std::vector<std::vector<double>> B = {};
  std::vector<std::vector<double>> result;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  MatrixMultiplicationTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(golovkin_rowwise_matrix_partitioning, Multiply_LargeMatrices) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A(100, std::vector<double>(100, 1.0));
  std::vector<std::vector<double>> B(100, std::vector<double>(100, 1.0));
  std::vector<std::vector<double>> result(100, std::vector<double>(100, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  MatrixMultiplicationTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());

  std::vector<std::vector<double>> expected(100, std::vector<double>(100, 100.0));
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(golovkin_rowwise_matrix_partitioning, EmptyMatrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {};
  std::vector<std::vector<double>> B = {{1.0, 2.0}, {3.0, 4.0}};

  std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->inputs_count = {sizeof(std::vector<std::vector<double>>), sizeof(std::vector<std::vector<double>>)};

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(std::vector<std::vector<double>>));

  MatrixMultiplicationTask task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(golovkin_rowwise_matrix_partitioning, IncompatibleMatrixSizes) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A = {{1.0, 2.0}, {3.0, 4.0}};
  std::vector<std::vector<double>> B = {{5.0, 6.0}};

  std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->inputs_count = {sizeof(std::vector<std::vector<double>>), sizeof(std::vector<std::vector<double>>)};

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(std::vector<std::vector<double>>));

  MatrixMultiplicationTask task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(golovkin_rowwise_matrix_partitioning, InvalidMatrixData) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> A;
  std::vector<std::vector<double>>* B = nullptr;

  std::vector<std::vector<double>> result(2, std::vector<double>(2, 0.0));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskData->inputs_count = {sizeof(std::vector<std::vector<double>>), sizeof(std::vector<std::vector<double>>)};

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(std::vector<std::vector<double>>));

  MatrixMultiplicationTask task(taskData);

  ASSERT_FALSE(task.validation());
}