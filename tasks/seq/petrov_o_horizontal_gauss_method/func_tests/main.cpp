#include <gtest/gtest.h>

#include <memory>
#include <numeric>

#include "seq/petrov_o_horizontal_gauss_method/include/ops_seq.hpp"

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_Simple) {
  size_t n = 3;
  std::vector<double> input_matrix = {2, 1, 0, -3, -1, 2, 0, 1, 2};
  std::vector<double> input_b = {8, -11, -3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], 8);
  ASSERT_DOUBLE_EQ(output[1], -8);
  ASSERT_DOUBLE_EQ(output[2], 2.5);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_IdentityMatrix) {
  size_t n = 3;
  std::vector<double> input_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> input_b = {1, 2, 3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], 1);
  ASSERT_DOUBLE_EQ(output[1], 2);
  ASSERT_DOUBLE_EQ(output[2], 3);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_NegativeValues) {
  size_t n = 3;
  std::vector<double> input_matrix = {-2, -1, -1, -1, -1, -1, -1, -2, -1};
  std::vector<double> input_b = {-1, -2, -3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], -1);
  ASSERT_DOUBLE_EQ(output[1], 1);
  ASSERT_DOUBLE_EQ(output[2], 2);
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_EmptyMatrix) {
  size_t n = 0;
  std::vector<double> input_matrix;
  std::vector<double> input_b;
  std::vector<double> output;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(petrov_o_horizontal_gauss_method_seq, TestGauss_ZeroDiagonal) {
  size_t n = 3;
  std::vector<double> input_matrix = {1, 1, 2, 1, 0, 1, 1, 1, 1};
  std::vector<double> input_b = {1, 2, 3};
  std::vector<double> output(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(n);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_b.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(n * sizeof(double));

  petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_DOUBLE_EQ(output[0], 4);
  ASSERT_DOUBLE_EQ(output[1], 1);
  ASSERT_DOUBLE_EQ(output[2], -2);
}