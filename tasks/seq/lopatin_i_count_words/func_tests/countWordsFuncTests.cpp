#include <gtest/gtest.h>

#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

TEST(lopatin_i_count_words_seq, test_empty_string) {
  std::string input = "";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), false);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(0, out[0]);
}

TEST(lopatin_i_count_words_seq, test_single_word) {
  std::string input = "Hello";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(1, out[0]);
}

TEST(lopatin_i_count_words_seq, test_multiple_words) {
  std::string input = "This is a test sentence";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(5, out[0]);
}

TEST(lopatin_i_count_words_seq, test_multiple_sentences) {
  std::string input = "This is a test sentence. This is another one. And one more. And another one.";
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  lopatin_i_count_words_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(15, out[0]);
}