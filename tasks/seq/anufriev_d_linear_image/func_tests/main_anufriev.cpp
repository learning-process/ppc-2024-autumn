#include <gtest/gtest.h>

#include <numeric>
#include <random>

#include "core/task/include/task.hpp"
#include "seq/anufriev_d_linear_image/include/ops_seq_anufriev.hpp"

static void gaussian_filter_seq(const std::vector<int>& input, int rows, int cols, std::vector<int>& output) {
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  output.assign(rows * cols, 0);

  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      int sum = 0;
      for (int kr = -1; kr <= 1; ++kr) {
        for (int kc = -1; kc <= 1; ++kc) {
          sum += input[(r + kr) * cols + (c + kc)] * kernel[kr + 1][kc + 1];
        }
      }
      output[r * cols + c] = sum / 16;
    }
  }
}

static std::vector<int> generate_test_image(int rows, int cols) {
  std::vector<int> image(rows * cols);
  std::iota(image.begin(), image.end(), 0);
  return image;
}

static std::vector<int> generate_random_image(int rows, int cols, int seed = 123) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> image(rows * cols);
  for (auto& val : image) val = dist(gen);
  return image;
}

TEST(anufriev_d_linear_image_func_seq, TestGaussianFilterSmall) {
  int rows = 5;
  int cols = 5;
  std::vector<int> input = generate_test_image(rows, cols);
  std::vector<int> expected_output;
  std::vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq(input, rows, cols, expected_output);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  anufriev_d_linear_image::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(anufriev_d_linear_image_func_seq, TestGaussianFilterMedium) {
  int rows = 10;
  int cols = 8;
  std::vector<int> input = generate_test_image(rows, cols);
  std::vector<int> expected_output;
  std::vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq(input, rows, cols, expected_output);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  anufriev_d_linear_image::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(anufriev_d_linear_image_func_seq, TestGaussianFilterRandom) {
  int rows = 12;
  int cols = 12;
  std::vector<int> input = generate_random_image(rows, cols);
  std::vector<int> expected_output;
  std::vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq(input, rows, cols, expected_output);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  anufriev_d_linear_image::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(anufriev_d_linear_image_func_seq, TestGaussianFilterOddDimensions) {
  int rows = 7;
  int cols = 5;
  std::vector<int> input = generate_test_image(rows, cols);
  std::vector<int> expected_output;
  std::vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq(input, rows, cols, expected_output);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  anufriev_d_linear_image::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}
