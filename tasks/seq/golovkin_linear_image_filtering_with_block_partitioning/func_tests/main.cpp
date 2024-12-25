// Golovkin Maksim Task#3

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_linear_image_filtering_with_block_partitioning/include/ops_seq.hpp"

using namespace std;

static void gaussian_filter_seq_block(const vector<int>& input, int rows, int cols, vector<int>& output,
                                      int block_size) {
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  output.assign(rows * cols, 0);

  for (int r = 0; r < rows; r += block_size) {
    for (int c = 0; c < cols; c += block_size) {
      for (int br = r; br < std::min(r + block_size, rows - 1); ++br) {
        for (int bc = c; bc < std::min(c + block_size, cols - 1); ++bc) {
          int sum = 0;
          for (int kr = -1; kr <= 1; ++kr) {
            for (int kc = -1; kc <= 1; ++kc) {
              if (br + kr >= 0 && br + kr < rows && bc + kc >= 0 && bc + kc < cols) {
                sum += input[(br + kr) * cols + (bc + kc)] * kernel[kr + 1][kc + 1];
              }
            }
          }
          output[br * cols + bc] = sum / 16;
        }
      }
    }
  }
}

static vector<int> generate_test_images(int rows, int cols) {
  vector<int> image(rows * cols);
  iota(image.begin(), image.end(), 0);
  return image;
}

static vector<int> generate_random_images(int rows, int cols, int seed = 123) {
  mt19937 gen(seed);
  uniform_int_distribution<int> dist(0, 255);
  vector<int> image(rows * cols);
  for (auto& val : image) val = dist(gen);
  return image;
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterSmall) {
  int rows = 5;
  int cols = 5;
  vector<int> input = generate_test_images(rows, cols);
  vector<int> expected_output;
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 2);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterRandom) {
  int rows = 10;
  int cols = 10;
  vector<int> input = generate_random_images(rows, cols);
  vector<int> expected_output;
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 4);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterSmallImage) {
  int rows = 3;
  int cols = 3;
  vector<int> input = generate_test_images(rows, cols);
  vector<int> expected_output = {1, 2, 3, 2, 4, 6, 3, 6, 9};
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterMaxPixelValues) {
  int rows = 5;
  int cols = 5;
  vector<int> input(rows * cols, 255);
  vector<int> expected_output(rows * cols, 255);
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterMinPixelValues) {
  int rows = 5;
  int cols = 5;
  vector<int> input(rows * cols, 0);
  vector<int> expected_output(rows * cols, 0);
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterSharpEdges) {
  int rows = 5;
  int cols = 5;
  vector<int> input = {0,   0,   0,   255, 255, 0, 0, 0,   255, 255, 0, 0, 0,
                       255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0};
  vector<int> expected_output = {1,   1,   2,   128, 128, 1, 2, 3,   128, 128, 2, 3, 4,
                                 128, 128, 128, 128, 128, 2, 2, 128, 128, 128, 3, 3};
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);
  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterRandomImage) {
  int rows = 10;
  int cols = 10;
  vector<int> input = generate_random_images(rows, cols);
  vector<int> expected_output(rows * cols, 0);
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterUniformImage) {
  int rows = 5;
  int cols = 5;
  vector<int> input(rows * cols, 128);            // Все пиксели равны 128
  vector<int> expected_output(rows * cols, 128);  // Ожидаемый результат также будет 128
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterImageWithNoise) {
  int rows = 5;
  int cols = 5;
  vector<int> input = {0, 255, 0, 0, 0, 0,   0, 255, 0,   0, 255, 0, 0,
                       0, 255, 0, 0, 0, 255, 0, 0,   255, 0, 0,   0};  // Ввод с шумом
  vector<int> expected_output = {
      32, 51, 32, 32, 32, 51, 85, 51, 32, 32, 32, 51, 32,
      51, 32, 32, 32, 51, 51, 32, 32, 32, 32, 32, 32};  // Ожидаемый результат может быть рассчитан
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterGradientImage) {
  int rows = 5;
  int cols = 5;
  vector<int> input = {0,   51,  102, 153, 204, 51,  102, 153, 204, 255, 102, 153, 204,
                       255, 255, 153, 204, 255, 255, 255, 204, 255, 255, 255, 255};  // Градиентное изображение
  vector<int> expected_output = {25,  51,  76,  102, 127, 51,  76,  102, 127, 153, 76,  102, 127,
                                 153, 178, 102, 127, 153, 178, 204, 127, 153, 178, 204, 229};  // Ожидаемый результат
  vector<int> actual_output(rows * cols, 0);

  gaussian_filter_seq_block(input, rows, cols, expected_output, 3);

  auto taskData = make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
  taskData->inputs_count.push_back(sizeof(int));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(actual_output.data()));
  taskData->outputs_count.push_back(actual_output.size());

  golovkin_linear_image_filtering_with_block_partitioning::SimpleIntSEQ task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(expected_output, actual_output);
}