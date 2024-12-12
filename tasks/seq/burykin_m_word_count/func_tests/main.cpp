#include <gtest/gtest.h>

#include "seq/burykin_m_word_count/include/ops_seq.hpp"

TEST(WordCountSequential, TestIsWordCharacter) {
  EXPECT_TRUE(burykin_m_word_count::TestTaskSequential::is_word_character('a'));
  EXPECT_TRUE(burykin_m_word_count::TestTaskSequential::is_word_character('Z'));

  EXPECT_TRUE(burykin_m_word_count::TestTaskSequential::is_word_character('\''));

  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('@'));
  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('#'));
  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('!'));
  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('$'));

  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character(' '));
  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('\n'));
  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('\t'));

  EXPECT_FALSE(burykin_m_word_count::TestTaskSequential::is_word_character('\0'));
}

TEST(WordCountSequential, EmptyString) {
  std::string input;
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(in.data());
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  burykin_m_word_count::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(0, out[0]);
}

TEST(WordCountSequential, SingleWord) {
  std::string input = "Hello.";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(in.data());
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  burykin_m_word_count::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(1, out[0]);
}

TEST(WordCountSequential, MultipleWords) {
  std::string input = "This is a test sentence.";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(in.data());
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  burykin_m_word_count::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(5, out[0]);
}

TEST(WordCountSequential, WordsWithApostrophes) {
  std::string input = "It's a beautiful day, isn't it?";
  std::vector<uint8_t> in(input.begin(), input.end());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(in.data());
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  burykin_m_word_count::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(6, out[0]);
}
