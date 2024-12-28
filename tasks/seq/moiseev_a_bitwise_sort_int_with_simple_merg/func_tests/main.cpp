#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"

template <typename DataType>
static std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = static_cast<int>(gen()) % 100;
  }
  return vec;
}

TEST(moiseev_a_radix_merge_seq_test, test_fixed_values) {
  std::vector<int> input = {3, -1, 4, 1, -5, 9, 2, 6, -5, 3, 5};
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_random_small_vector) {
  int size = 100;
  std::vector<int> input = getRandomVector<int>(size);
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_random_large_vector) {
  int size = 1000000;
  std::vector<int> input = getRandomVector<int>(size);
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_empty_array) {
  std::vector<int> input = {};
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_single_element) {
  std::vector<int> input = {42};
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_negative_numbers) {
  std::vector<int> input = {-3, -1, -4, -1, -5, -9, -2, -6, -5, -3, -5};
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}

TEST(moiseev_a_radix_merge_seq_test, test_random_medium_vector) {
  int size = 10000;
  std::vector<int> input = getRandomVector<int>(size);
  std::vector<int> output(input.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(output.size());

  moiseev_a_radix_merge_seq::TestSEQTaskSequential task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(output, expected);
}
