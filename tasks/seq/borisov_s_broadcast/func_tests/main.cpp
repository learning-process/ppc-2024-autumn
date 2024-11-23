#include <gtest/gtest.h>

#include <vector>

#include "seq/borisov_s_broadcast/include/ops_seq.hpp"

using namespace borisov_s_broadcast;

TEST(DistanceMatrixTask, ValidInput) {
  std::vector<double> points = {0.0, 0.0, 3.0, 4.0, 6.0, 8.0};

  size_t n = points.size() / 2;
  std::vector<double> expected_matrix = {0.0, 5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0};

  ppc::core::TaskData taskData;
  taskData.inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskData.inputs_count.emplace_back(n);
  taskData.outputs.resize(1);
  taskData.outputs_count.emplace_back(n * n);
  taskData.outputs[0] = reinterpret_cast<uint8_t*>(new double[n * n]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(std::make_shared<ppc::core::TaskData>(taskData));

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto* output = reinterpret_cast<double*>(taskData.outputs[0]);
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(output[i], expected_matrix[i], 1e-6);
  }

  delete[] output;
}

TEST(DistanceMatrixTask, EmptyInput) {
  ppc::core::TaskData taskData;

  taskData.inputs_count.emplace_back(0);
  taskData.outputs_count.emplace_back(0);
  taskData.outputs.emplace_back(nullptr);
  taskData.inputs.emplace_back(nullptr);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(std::make_shared<ppc::core::TaskData>(taskData));

  ASSERT_FALSE(task->validation());
}

TEST(DistanceMatrixTask, MissingOutputBuffer) {
  std::vector<double> points = {0.0, 0.0, 3.0, 4.0};

  ppc::core::TaskData taskData;
  taskData.inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskData.inputs_count.emplace_back(2);
  taskData.outputs_count.emplace_back(0);
  taskData.outputs.emplace_back(nullptr);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(std::make_shared<ppc::core::TaskData>(taskData));

  ASSERT_FALSE(task->validation());
}

TEST(DistanceMatrixTask, SinglePoint) {
  std::vector<double> points = {1.0, 2.0};

  ppc::core::TaskData taskData;
  taskData.inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskData.inputs_count.emplace_back(1);
  taskData.outputs.resize(1);
  taskData.outputs_count.emplace_back(1);
  taskData.outputs[0] = reinterpret_cast<uint8_t*>(new double[1]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(std::make_shared<ppc::core::TaskData>(taskData));

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto* output = reinterpret_cast<double*>(taskData.outputs[0]);
  EXPECT_NEAR(output[0], 0.0, 1e-6);

  delete[] output;
}

TEST(DistanceMatrixTask, LargeInput) {
  const int point_count = 1000;
  auto points = getRandomPoints(point_count);

  ppc::core::TaskData taskData;
  taskData.inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskData.inputs_count.emplace_back(point_count);
  taskData.outputs.resize(1);
  taskData.outputs_count.emplace_back(point_count * point_count);
  taskData.outputs[0] = reinterpret_cast<uint8_t*>(new double[point_count * point_count]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(std::make_shared<ppc::core::TaskData>(taskData));

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto* output = reinterpret_cast<double*>(taskData.outputs[0]);
  for (size_t i = 0; i < point_count; i++) {
    EXPECT_GE(output[(i * point_count) + i], 0.0);
  }

  delete[] output;
}