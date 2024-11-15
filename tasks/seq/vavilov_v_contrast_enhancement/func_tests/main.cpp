#include <gtest/gtest.h>

#include <vector>

#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

TEST(vavilov_v_contrast_enhancement_seq, ValidInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->inputs[0] = input.data();

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
}

TEST(vavilov_v_contrast_enhancement_seq, EmptyInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count[0] = 0;

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
}

TEST(vavilov_v_contrast_enhancement_seq, ValidOutputSize) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size();

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
}

TEST(vavilov_v_contrast_enhancement_seq, MismatchedOutputSize) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size() - 1;

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_FALSE(task.validation());
}

TEST(vavilov_v_contrast_enhancement_seq, NormalContrastEnhancement) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size();
  std::vector<int> output(input.size());
  taskData->inputs[0] = input.data();
  taskData->outputs[0] = output.data();

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());

  std::vector<int> expected_output = {0, 63, 127, 191, 255};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_contrast_enhancement_seq, SingleValueInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {100, 100, 100};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size();
  std::vector<int> output(input.size());
  taskData->inputs[0] = input.data();
  taskData->outputs[0] = output.data();

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());

  std::vector<int> expected_output(input.size(), 0);
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_contrast_enhancement_seq, ValidOutputCopy) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskData->inputs_count[0] = input.size();
  taskData->outputs_count[0] = input.size();
  std::vector<int> output(input.size());
  taskData->inputs[0] = input.data();
  taskData->outputs[0] = output.data();

  vavilov_v_contrast_enhancement_seq::ContrastEnhancementSequential task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<int> expected_output = {0, 63, 127, 191, 255};
  EXPECT_EQ(output, expected_output);
}
