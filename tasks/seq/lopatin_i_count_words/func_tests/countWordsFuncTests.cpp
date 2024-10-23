#include <gtest/gtest.h>

#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

TEST(lopatin_i_count_words_seq, test_300_words) {
  std::vector<char> input = lopatin_i_count_words_seq::generateLongString(20);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 300);
}

TEST(lopatin_i_count_words_seq, test_1500_words) {
  std::vector<char> input = lopatin_i_count_words_seq::generateLongString(100);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 1500);
}

TEST(lopatin_i_count_words_seq, test_6k_words) {
  std::vector<char> input = lopatin_i_count_words_seq::generateLongString(400);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 6000);
}