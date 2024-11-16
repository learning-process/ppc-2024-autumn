#include <gtest/gtest.h>

#include <vector>

#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

TEST(vavilov_v_contrast_enhancement_seq, ValidInput) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<int> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
}

TEST(vavilov_v_contrast_enhancement_seq, EmptyInput) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(0);

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(vavilov_v_contrast_enhancement_seq, ValidOutputSize) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(input.size());

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
}

TEST(vavilov_v_contrast_enhancement_seq, MismatchedOutputSize) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs_count.emplace_back(input.size() - 1);

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(vavilov_v_contrast_enhancement_seq, NormalContrastEnhancement) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());
  std::vector<int> output(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());

  std::vector<int> expected_output = {0, 63, 127, 191, 255};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_contrast_enhancement_seq, SingleValueInput) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {100, 100, 100};
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());
  std::vector<int> output(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());

  std::vector<int> expected_output(input.size(), 0);
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_contrast_enhancement_seq, ValidOutputCopy) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input = {10, 20, 30, 40, 50};
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());
  std::vector<int> output(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, 63, 127, 191, 255};
  EXPECT_EQ(output, expected_output);
}
